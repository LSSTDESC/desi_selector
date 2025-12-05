import opencosmo as oc
import numpy as np
import pandas as pd
from pathlib import Path

# data_path = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/")
data_path = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/smdpl_dr1_2025_11_07/")
file_list = list(data_path.glob("*.hdf5"))

tracer_type = 'elg'
sfh_model = 'um'
output_file_name = f'{tracer_type}_sim_cat_colors_{sfh_model}_sfh.hdf5'
output_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/sim_data/{tracer_type}_1000deg2/{output_file_name}'


def save_sim_cat(file_list=file_list, output_path=output_path):

    dataset = oc.open(file_list)
    columns = ['ra', 'dec', 'redshift_true', 'logsm_obs', 'logssfr_obs',
               'lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y'
              ]
    dataset = dataset.select(columns)
    oc.write(output_path, dataset, overwrite=True)
    
save_sim_cat()

            




# if __name__ == '__main__':


#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     comm.Barrier()

#     for f_name in gal_file_list[rank::size]:
#         save_sim_cat(f_name)

 

 