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


Z_GRID_POINTS = 481
ELG_TARG_DENS_AVG = (1930 + 1950 + 1900) / 3
RAND_TO_DATA_RATIO = 10

sfh_model = 'um'
path_log_sfr_thres = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/selection_thresholds/elg/sfr_thres_{sfh_model}_sfh.npy'
dataset_path = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/sim_data/elg_1000deg2/elg_sim_cat_{sfh_model}_sfh.hdf5'

log_sfr_thres = np.load(path_log_sfr_thres)

# load in the desi ELG distributions 
path_desi_data = '/pscratch/sd/y/yoki/desi_like_data_diffsky/data/desi_sv_data/desi_elg_ts_zenodo/main-800coaddefftime1200-nz-zenodo.ecsv'
desi_data = table.Table.read(path_desi_data, format='ascii.ecsv')
zmin = desi_data['ZMIN']
zmax = desi_data['ZMAX']

new_zmin = np.linspace(np.min(zmin), np.max(zmax), Z_GRID_POINTS)[:-1]
new_zmax = np.linspace(np.min(zmin), np.max(zmax), Z_GRID_POINTS)[1:]
new_z_center = (new_zmax + new_zmin) / 2
thres_of_z = interpolate.interp1d(new_z_center, log_sfr_thres,  fill_value=9E11, bounds_error=False)


def save_mock_elg_cat(dataset_path=dataset_path):

    dataset = oc.open(dataset_path)
    sim_cat = dataset.data.to_pandas()
    sim_cat['distance'] = cosmo.comoving_distance(sim_cat['redshift_true']).value

    
    threshold_all = thres_of_z(sim_cat['redshift_true'])
    mask_sfr = sim_cat['log_sfr'] > threshold_all
    mock_elg_cat = sim_cat[mask_sfr]
    
 
    npatches = 40
    ntot = int(len(mock_elg_cat)* RAND_TO_DATA_RATIO / npatches)
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
    mock_elg_cat_temp = mock_elg_cat.reset_index(drop=True).sample(len(rand_cat), replace=True)
    rand_cat['distance'] = mock_elg_cat_temp['distance'].to_numpy()
    rand_cat['redshift_true'] = mock_elg_cat_temp['redshift_true'].to_numpy()


    rand_output_file_name = f'rand_elg_cat_{sfh_model}_sfh.parquet'
    rand_cat_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/mock_cats/elg/{rand_output_file_name}'
    rand_cat.to_parquet(rand_cat_path)

    
    output_file_name = f'mock_elg_cat_{sfh_model}_sfh.parquet'
    mock_elg_cat_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/mock_cats/elg/{output_file_name}'
    mock_elg_cat.to_parquet(mock_elg_cat_path)
    



save_mock_elg_cat()
           




# if __name__ == '__main__':


#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     comm.Barrier()

#     for f_name in gal_file_list[rank::size]:
#         save_mock_elg_cat(f_name)

 

 