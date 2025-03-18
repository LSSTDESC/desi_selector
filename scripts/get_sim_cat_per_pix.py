import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import table
import h5py
import healpy as hp
import glob
import sys
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import fastparquet

path_healpix_ids = '/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/healpix_ids/id_nums_exclude_edges.npy'
healpix_ids = np.load(path_healpix_ids)

NEAR_0 = 3
NEAR_360 = 357
G_MAG_CUT = 26



def save_sim_cat_hpix(healpix_id=None):
    # print(f'{worker_index} is processing {healpix_id}')
    

    cat_list = []
    z_range_skysim = [[0,1], [1,2]]

    for z in z_range_skysim:
        
        filepath = '/global/cfs/cdirs/lsst/shared/xgal/skysim/skysim5000_v1.1.1'
        h5_filename = f'/z_{z[0]}_{z[1]}.step_all.healpix_{healpix_id}.hdf5' # assuming all healpix files have same root file 
        h5f = filepath + h5_filename
    
    
        with h5py.File(h5f, 'r') as file:
            
         
            
            properties = file['galaxyProperties']
            redshift = np.array(properties['redshift'])
            # r = cosmo.comoving_distance(redshift).value # units of Mpc
            sfr = np.array(properties['baseDC2']['sfr'])
            sfr_tot = np.array(properties['totalStarFormationRate'])
            # stellar_mass = np.array(properties['totalMassStellar'])
            # blackhole_mass = np.array(properties['blackHoleMass'])
            # gal_id = np.array(properties['galaxyID'])
            # mag_u = np.array(properties['LSST_filters']['magnitude:LSST_u:observed:dustAtlas']) # mags with no MW extinction corrections
            mag_g = np.array(properties['LSST_filters']['magnitude:LSST_g:observed:dustAtlas'])
            # mag_r = np.array(properties['LSST_filters']['magnitude:LSST_r:observed:dustAtlas'])
            # mag_i = np.array(properties['LSST_filters']['magnitude:LSST_i:observed:dustAtlas'])
            # mag_z = np.array(properties['LSST_filters']['magnitude:LSST_z:observed:dustAtlas'])
            # mag_y = np.array(properties['LSST_filters']['magnitude:LSST_y:observed:dustAtlas'])
            ra = np.array(properties['ra'])
            # dec = np.array(properties['dec'])
            # ra_true = np.array(properties['ra_true'])
            # dec_true = np.array(properties['dec_true'])


            array_list = np.column_stack([redshift, sfr, sfr_tot, ra, mag_g])

            sim_cat_in = pd.DataFrame(array_list, columns=['redshift', 'sfr','sfr_tot', 'ra', 'mag_g'])
            elg_mag_cut = sim_cat_in['mag_g'] < G_MAG_CUT
            edge_mask = np.logical_and(sim_cat_in['ra'] < NEAR_360, sim_cat_in['ra'] > NEAR_0)
            masks = np.logical_and(elg_mag_cut, edge_mask)
            cat_list.append(sim_cat_in[masks])

        sim_cat = pd.concat(cat_list)
        output_file_name = f'sim_cat_hpix_{healpix_id}'
        output = f'{output_file_name}.parquet'
        sim_cat_path = f'/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/sim_data/{output}'
        sim_cat.to_parquet(sim_cat_path)
        




if __name__ == '__main__':


    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()

    for pixel_id in healpix_ids[rank::size]:
        save_sim_cat_hpix(pixel_id)

 

 