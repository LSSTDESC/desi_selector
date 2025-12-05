import opencosmo as oc
import numpy as np
import pandas as pd
from pathlib import Path


def log_sfr(logsm_obs, logssfr_obs):
    return logsm_obs + logssfr_obs


def save_sim_cat(file_list, path_output, z_range=[0,2]):

    dataset = oc.open(file_list)
    columns = ['ra', 'dec', 'redshift_true', 'logsm_obs', 'logmp_obs', 'lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y']
    
    dataset = dataset.select(columns)
    dataset = dataset.with_redshift_range(z_range[0], z_range[1])
    
    dataset = dataset.evaluate(log_sfr, vectorize=True)
    oc.write(path_output, dataset, overwrite=True)

tracer_type = 'elg'
dict_mock_paths = {'tng_2025_11_06': 'tng', 'smdpl_dr1_2025_11_07': 'um', 'galacticus_in_plus_ex_situ_2025_11_10': 'gal'}


for key, value in dict_mock_paths.items():
    
    path_data = Path(f"/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/{key}/")
    file_list = list(path_data.glob("*.hdf5"))
    output_file_name = f'{tracer_type}_sim_cat_{value}_sfh.hdf5'
    path_output = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/sim_data/{tracer_type}_1000deg2/{output_file_name}'
    save_sim_cat(file_list=file_list, path_output=path_output)

            





# if __name__ == '__main__':


#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     comm.Barrier()

#     for f_name in gal_file_list[rank::size]:
#         save_sim_cat(f_name)

 

 