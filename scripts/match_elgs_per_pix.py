import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal
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

# 
path_healpix_ids = '/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/healpix_ids/id_nums_exclude_edges.npy'
path_sfr_thres = '/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/selection_thresholds/elg_sfr_thres_1000bins.npy'
healpix_ids = np.load(path_healpix_ids)
sfr_thres = np.load(path_sfr_thres)

LOP_NORTH_AREA = 4400
LOP_SOUTH_DECAL_AREA = 8500
LOP_SOUTH_DES_AREA = 1100
TOTAL_DESI_AREA = 14000
DESI_ELG_ZBIN_WIDTH = 0.05
Z_GRID_POINTS = 1001
AREA_PER_HEALPIX = 57.071968/17 # Area of 17 healpix divided by 17
ELG_TARG_DENS_AVG = (1930 + 1950 + 1900) / 3
LOP_ELG_MAG_CUTOFF = 26
NEAR_0 = 3
NEAR_360 = 357
NSIDE = 32
NUM_POINTS = int(ELG_TARG_DENS_AVG * AREA_PER_HEALPIX * 40) # overkill to make sure we have more than enough points
DIST_WITHIN_SIM = 1/2 # arcmin
RAND_TO_DATA_RATIO = 10


# load in the desi ELG distributions 
path_desi_data = '/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/desi_sv_data/desi_elg_ts_zenodo/main-800coaddefftime1200-nz-zenodo.ecsv'
desi_data = table.Table.read(path_desi_data, format='ascii.ecsv')
zmin = desi_data['ZMIN']
zmax = desi_data['ZMAX']

new_zmin = np.linspace(np.min(zmin), np.max(zmax), Z_GRID_POINTS)[:-1]
new_zmax = np.linspace(np.min(zmin), np.max(zmax), Z_GRID_POINTS)[1:]
new_z_center = (new_zmax + new_zmin)/2
# z_bin_centers = (zmin + zmax ) / 2 
# smooth_thres = signal.convolve(sfr_thres, np.array([1/5, 1/5, 1/5, 1/5, 1/5]), mode='same')
thres_of_z = interpolate.interp1d(new_z_center, sfr_thres,  fill_value=9E11, bounds_error=False)


def save_mock_elg_sample(healpix_id=None):
    
    cat_list = []
    z_range_skysim = [[0,1], [1,2]]

    
    for z in z_range_skysim:

        filepath = f'/global/cfs/cdirs/lsst/shared/xgal/skysim/skysim5000_v1.1.1/z_{z[0]}_{z[1]}.step_all.healpix_{healpix_id}.hdf5'
    
        with h5py.File(filepath, 'r') as file:
         
            
            properties = file['galaxyProperties']
            redshift = np.array(properties['redshift'])
            redshift_hubble = np.array(properties['redshiftHubble'])
            distance = cosmo.comoving_distance(redshift).value
            distance_hubble = cosmo.comoving_distance(redshift_hubble).value
            sfr = np.array(properties['baseDC2']['sfr'])
            sfr_tot = np.array(properties['totalStarFormationRate'])
            stellar_mass = np.array(properties['totalMassStellar'])
            blackhole_mass = np.array(properties['blackHoleMass'])
            gal_id = np.array(properties['galaxyID'])
            mag_u = np.array(properties['LSST_filters']['magnitude:LSST_u:observed:dustAtlas']) # mags with no MW extinction corrections
            mag_g = np.array(properties['LSST_filters']['magnitude:LSST_g:observed:dustAtlas'])
            mag_r = np.array(properties['LSST_filters']['magnitude:LSST_r:observed:dustAtlas'])
            mag_i = np.array(properties['LSST_filters']['magnitude:LSST_i:observed:dustAtlas'])
            mag_z = np.array(properties['LSST_filters']['magnitude:LSST_z:observed:dustAtlas'])
            mag_y = np.array(properties['LSST_filters']['magnitude:LSST_y:observed:dustAtlas'])
            ra = np.array(properties['ra'])
            dec = np.array(properties['dec'])
            ra_true = np.array(properties['ra_true'])
            dec_true = np.array(properties['dec_true'])


            array_list = np.column_stack([redshift, redshift_hubble, distance, distance_hubble, sfr, sfr_tot, stellar_mass, blackhole_mass, gal_id, mag_u,
                                        mag_g, mag_r, mag_i, mag_z, mag_y
                                          , ra, dec, ra_true, dec_true])

            sim_cat_in = pd.DataFrame(array_list, columns=['redshift', 'redshift_hubble', 'distance', 'distance_hubble', 'sfr','sfr_tot','stellar_mass','blackhole_mass','gal_id','mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'mag_y', 'ra', 'dec', 'ra_true', 'dec_true'])
            edge_mask = np.logical_and(sim_cat_in['ra'] < NEAR_360, sim_cat_in['ra'] > NEAR_0)
            cat_list.append(sim_cat_in[edge_mask])
    
            

        sim_cat = pd.concat(cat_list)
        
        # print(f'the number of galaxies in this cat is {len(sim_cat)}')
        
        ra_min = np.min(ra[edge_mask])
        ra_max = np.max(ra[edge_mask])
        
        rand_ra = ra_min + (ra_max - ra_min)*np.random.random(size=NUM_POINTS)
        cth_min = np.min(np.sin(np.radians(sim_cat['dec'])))
        cth_max = np.max(np.sin(np.radians(sim_cat['dec'])))
        cth_rand = cth_min + (cth_max - cth_min)*np.random.random(size=NUM_POINTS)
        rand_dec = np.degrees(np.arcsin(cth_rand))
        
        # Convert to HEALPix θ (colatitude) and φ (longitude)
        rand_theta = np.radians(90.0 - rand_dec)  
        rand_phi = np.radians(rand_ra)            
        
        # Get HEALPix pixel id
        rand_pix_id = hp.ang2pix(NSIDE, rand_theta, rand_phi, nest=False) 
        rand_hpix_mask = rand_pix_id == healpix_id
        
        
        rand_cols_list = np.column_stack([rand_ra[rand_hpix_mask], rand_dec[rand_hpix_mask]])
        rand_cat = pd.DataFrame(rand_cols_list, columns=['ra', 'dec'])



        elg_theta = np.radians(90.0 - sim_cat['dec'])  
        elg_phi = np.radians(sim_cat['ra'])            
        
        # Get HEALPix pixel id
        elg_pix_id = hp.ang2pix(NSIDE, elg_theta, elg_phi, nest=False) 
        elg_hpix_mask = elg_pix_id == healpix_id
        
        
        sim_cat_gmag_mask = sim_cat['mag_g'] < LOP_ELG_MAG_CUTOFF
        elg_masks = np.logical_and(sim_cat_gmag_mask, elg_hpix_mask)
        sim_cat_masked = sim_cat[elg_masks]
        
        
        threshold_all = thres_of_z(sim_cat_masked['redshift'])
        sfr_mask = sim_cat_masked['sfr_tot'] > threshold_all
        mock_elg_cat = sim_cat_masked[sfr_mask]

        # add distances and redshift to random catalog
        rand_cat = rand_cat.reset_index(drop=True) 
        mock_elg_cat_temp = mock_elg_cat.reset_index(drop=True).sample(len(rand_cat), replace=True)


        rand_cat['distance'] = mock_elg_cat_temp['distance'].to_numpy()
        rand_cat['distance_hubble'] = mock_elg_cat_temp['distance_hubble'].to_numpy()
        rand_cat['redshift'] = mock_elg_cat_temp['redshift'].to_numpy()
        
        # mask the random cat to have around ten times as many objects as the mock elg cat
        NUM_TO_KEEP = len(mock_elg_cat) * RAND_TO_DATA_RATIO # number of objects to keep for randoms
        rand_cat_final = rand_cat.iloc[:NUM_TO_KEEP]

        rand_output_file_name = f'rand_elg_cat_hpix_{healpix_id}'
        rand_output = f'{rand_output_file_name}.parquet'
        rand_cat_path = f'/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/rands_per_healpix/rands_elg_cat_per_pixel/{rand_output}'
        rand_cat_final.to_parquet(rand_cat_path)
        
        output_file_name = f'mock_elg_cat_hpix_{healpix_id}'
        output = f'{output_file_name}.parquet'
        mock_elg_cat_path = f'/global/homes/y/yoki/roman/desi_like_samples/skysim_5000/data/mocks_per_healpix/mock_elg_cat_per_pixel/{output}'
        mock_elg_cat.to_parquet(mock_elg_cat_path)


if __name__ == '__main__':


    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()

    for pixel_id in healpix_ids[rank::size]:
        save_mock_elg_sample(pixel_id)

 


 