import opencosmo as oc
import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import table
import h5py
from pathlib import Path
from astropy.cosmology import LambdaCDM

from diffsky.experimental import lc_utils
from diffsky.data_loaders.hacc_utils import lightcone_utils
import jax.random as jran

# Cosmology used in the simulation
omega_c = 0.26067
omega_b = 0.049
h = 0.6766
n_s = 0.9665
sigma8 = 0.8102
cosmo = LambdaCDM(H0=h * 100, Om0=omega_c + omega_b, Ode0=1 - (omega_c + omega_b))
RAND_TO_DATA_RATIO = 10
Z_GRID_POINTS = 481

sfh_model = 'galacticus'
selection_property = 'luminosity'


path_dataset = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/sim_data/qso_1000deg2/qso_sim_cat_{sfh_model}_sfh.hdf5'
path_lum_thres = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/selection_thresholds/qso/{selection_property}_thres_{sfh_model}_sfh.npy'
path_qso_data = '/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/desi_sv_data/desi_qso_ts_zenodo/fig_20_data.ecsv'

qso_data = table.Table.read(path_qso_data, format='ascii.ecsv')
nz_avg_original = (qso_data['n_z_north'] + qso_data['n_z_north']) / 2
lum_thres = np.load(path_lum_thres, allow_pickle=True)

z_mask = np.logical_and(qso_data['z'] > 0, qso_data['z'] < 3)
z_bin_center = qso_data['z'][z_mask]
z_bin_min = z_bin_center - 0.050/2
z_bin_max = z_bin_center + 0.050/2

nz_north = qso_data['n_z_north'][z_mask]
nz_south = qso_data['n_z_south'][z_mask]
nz_avg = (nz_north + nz_south) / 2 

zgrid = np.linspace(np.min(z_bin_min), np.max(z_bin_max), Z_GRID_POINTS)
new_z_bin_min = np.linspace(np.min(z_bin_min), np.max(z_bin_max), Z_GRID_POINTS)[:-1]
new_z_bin_max = np.linspace(np.min(z_bin_min), np.max(z_bin_max), Z_GRID_POINTS)[1:]
new_z_center = (new_z_bin_max + new_z_bin_min) / 2
thres_of_z = interpolate.interp1d(new_z_center, lum_thres, fill_value=20E11, bounds_error=False)



def save_mock_cat(path_dataset=path_dataset):

    dataset = oc.open(dataset_path)
    sim_cat = dataset.data.to_pandas()
    sim_cat['distance'] = cosmo.comoving_distance(sim_cat['redshift_true']).value
    
    threshold_all = thres_of_z(sim_cat['redshift_true'])
    mask_lum = sim_cat['black_hole_mass']*sim_cat['black_hole_eddington_ratio'] > threshold_all
    mock_cat = sim_cat[mask_lum]

    npatches = 40
    ntot = int(len(mock_cat)* RAND_TO_DATA_RATIO / npatches)
    lc_path = '/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/lc_metadata/lc_cores-decomposition.txt'
    lc_cores_decomp = lightcone_utils.read_lc_ra_dec_patch_decomposition(lc_path)[0]
    theta_low = lc_cores_decomp[:,1]
    theta_high = lc_cores_decomp[:,2]
    phi_low = lc_cores_decomp[:,3]
    phi_high = lc_cores_decomp[:,4]


    ra_min, dec_max = lightcone_utils.get_ra_dec_from_theta_phi(theta_low, phi_low)
    ra_max, dec_min = lightcone_utils.get_ra_dec_from_theta_phi(theta_high, phi_high)
    ran_key = jran.PRNGKey(0)

    
    patches = np.arange(0,40)
    list_array_ra = []
    list_array_dec = []
    
    for patch in patches:
        
        ra_loop, dec_loop = lc_utils.mc_lightcone_random_ra_dec(ran_key=ran_key, npts=ntot, ra_min=ra_min[patch],
        ra_max=ra_max[patch], dec_min=dec_min[patch], dec_max=dec_max[patch])

        list_array_ra.append(ra_loop)
        list_array_dec.append(dec_loop)
                                    
        
    rand_ra = np.concatenate(list_array_ra)
    rand_dec = np.concatenate(list_array_dec)
    rand_cols_list = np.column_stack([rand_ra, rand_dec])
    rand_cat = pd.DataFrame(rand_cols_list, columns=['ra', 'dec'])
    rand_cat = rand_cat.reset_index(drop=True) 
    mock_cat_temp = mock_cat.reset_index(drop=True).sample(len(rand_cat), replace=True)
    rand_cat['distance'] = mock_cat_temp['distance'].to_numpy()
    rand_cat['redshift_true'] = mock_cat_temp['redshift_true'].to_numpy()


    rand_output_file_name = f'rand_qso_cat_{selection_property}_{sfh_model}_sfh.parquet'
    rand_cat_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/mock_cats/qso/{rand_output_file_name}'
    rand_cat.to_parquet(rand_cat_path)

    
    output_file_name = f'mock_qso_cat_{selection_property}_{sfh_model}_sfh.parquet'
    mock_cat_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/mock_cats/qso/{output_file_name}'
    mock_cat.to_parquet(mock_cat_path)
    



save_mock_cat()
           



 