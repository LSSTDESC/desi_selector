import opencosmo as oc
import numpy as np
import pandas as pd
from astropy import table
import h5py
from pathlib import Path
from astropy.cosmology import LambdaCDM
from diffsky.experimental import lc_utils
from diffsky.data_loaders.hacc_utils import lightcone_utils
import jax.random as jran

COLOR_CUT1_SLOPE = 0.50
COLOR_CUT1_INTERCEPT = 0.1
COLOR_CUT2_SLOPE = -1.20
COLOR_CUT2_INTERCEPT = 1.3
LOP_ELG_MAG_CUTOFF = 24.1 - 0.65

# Cosmology used in the simulation
omega_c = 0.26067
omega_b = 0.049
h = 0.6766
n_s = 0.9665
sigma8 = 0.8102
cosmo = LambdaCDM(H0=h * 100, Om0=omega_c + omega_b, Ode0=1 - (omega_c + omega_b))
RAND_TO_DATA_RATIO = 10

tracer_type = 'elg'
sfh_model = 'um'
dataset_path = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/sim_data/{tracer_type}_1000deg2/{tracer_type}_sim_cat_colors_{sfh_model}_sfh.hdf5'


def save_mock_elg_cat(dataset_path=dataset_path):

    dataset = oc.open(dataset_path)
    sim_cat = dataset.data.to_pandas()
    sim_cat['distance'] = cosmo.comoving_distance(sim_cat['redshift_true']).value
    
    # DESI ELG color cuts: (g-r) < 0.50*(r-z) + 0.1 & g-r < -1.2*(r-z) + 1.3
    mask_color = np.logical_and((sim_cat['lsst_g'] - sim_cat['lsst_r']) < COLOR_CUT1_SLOPE*(sim_cat['lsst_r'] - sim_cat['lsst_z']) + COLOR_CUT1_INTERCEPT, (sim_cat['lsst_g'] - sim_cat['lsst_r']) < COLOR_CUT2_SLOPE*(sim_cat['lsst_r'] - sim_cat['lsst_z']) + COLOR_CUT2_INTERCEPT)
    mask_g_mag = sim_cat['lsst_g'] < LOP_ELG_MAG_CUTOFF
    masks_elg = np.logical_and(mask_color, mask_g_mag)
    mock_elg_cat = sim_cat[masks_elg]

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


    rand_output_file_name = f'rand_{tracer_type}_cat_colors_{sfh_model}_sfh.parquet'
    rand_cat_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/mock_cats/elg/{rand_output_file_name}'
    rand_cat.to_parquet(rand_cat_path)

    
    output_file_name = f'mock_{tracer_type}_cat_colors_{sfh_model}_sfh.parquet'
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

 

 