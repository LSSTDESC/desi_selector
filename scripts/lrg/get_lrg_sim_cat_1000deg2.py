import opencosmo as oc
import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies_1000deg2_unlensed/")
file_list = list(data_path.glob("*.hdf5"))

tracer_type = 'lrg'
output_file_name = f'{tracer_type}_sim_cat_1000deg2.hdf5'
output_path = f'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/sim_data/{tracer_type}_1000deg2/{output_file_name}'

def log_stellar_mass(logsm_obs):
    return logsm_obs 

def save_sim_cat(file_list=file_list, output_path=output_path, z_range=[0, 2]):

    dataset = oc.open(file_list)
    columns = ['ra', 'dec', 'redshift_true', 'logsm_obs', 'lsst_u', 'lsst_g', 'lsst_r', 'lsst_i', 'lsst_z', 'lsst_y'
              ]
    dataset = dataset.select(columns)
    dataset = dataset.with_redshift_range(z_range[0], z_range[1])
    dataset = dataset.evaluate(log_stellar_mass, vectorize=True)
    oc.write(output_path, dataset)
    
save_sim_cat()
 

 