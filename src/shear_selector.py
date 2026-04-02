import opencosmo as oc
import numpy as np
import pandas as pd
from scipy import interpolate
from pathlib import Path
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
# from diffsky.experimental import lc_utils
# from diffsky.data_loaders.hacc_utils import lightcone_utils
# import jax.random as jran
# import treecorr as tc
# import healpy as hp


class ShearSelector:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 threshold_col
                 lsst_year,
                 path_sim,
                 calibration_version,
                 z_range,
                 z_grid_points,
                 reload_oc
                 ):
        self.threshold_col = threshold_col
        self.lsst_year = lsst_year
        self.path_sim = path_sim
        self.calibration_version = calibration_version
        self.z_range = z_range
        self.z_grid_points = z_grid_points
        self.reload_oc = reload_oc

    

        
        path_sim_data = Path(f"{self.path_sim}/{calibration_version}")
        list_sim_data = list(f for f in path_sim_data.glob("*.hdf5") if f.stem.startswith("lc_cores"))
        dataset = oc.open(list_sim_data)

        # Calculate the total area the mocks span on the sky 
        dataset = oc.open(list_sim_data)
        pixels = dataset.region.pixels
        nside = dataset.region.nside
        sim_area = len(pixels)*hp.nside2pixarea(nside, degrees=True)
        self.sim_area = sim_area

        # Columns to keep from dataset
        columns = ['ra', 'dec', 'redshift_true', 'logsm_obs']

        print(f'The total area spanned by the mocks in {self.calibration_version} is: {self.sim_area}')


        if self.reload_oc:
        
            dataset = dataset.select(columns)
            dataset = dataset.with_redshift_range(self.z_range[0], self.z_range[1])
            sim_cat = dataset.get_data('pandas')
            sim_cat.rename(columns={'logsm_obs': 'log_stellar_mass'}, inplace=True)
            sim_cat['distance'] = ShearSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value
    
            sim_cat_filename = f"/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/sim_data/shear/{self.calibration_version}_{self.sim_area}.parquet"
            sim_cat.to_parquet(sim_cat_filename)
        else:
        
            sim_cat_filename = f"/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/sim_data/shear/{self.calibration_version}_{self.sim_area}.parquet"
            
            if Path(sim_cat_filename).exists():
                
                sim_cat = pd.read_parquet(sim_cat_filename)

        self.sim_cat = sim_cat

    def bin_shear_fit(self):

        
        if self.lsst_year == 1:

            
            # analytic fit to the LSST Y1 n(z) predictions for abundance matching
            NEFF = 11.1 # units of galaxies per square arcminute
            dz = (self.z_range[0] - self.z_range[1] )/ (self.z_grid_points - 1) 
            SQUARE_ARC_MIN_TO_SQUARE_DEG = 3600
            zgrid = np.linspace(self.z_range[0], self.z_range[0], self.z_grid_points)
            nz_fit =  (zgrid**2)*np.exp(-1*(zgrid/0.26)*(0.94))
            NORM = (NEFF) / np.sum(nz_fit)
            nz_fit = nz_fit * NORM * SQUARE_ARC_MIN_TO_SQUARE_DEG 
            zedges = np.append(zgrid, zgrid[-1] + dz ) - dz/2

            
        if self.lsst_year == 10:

            
            NEFF = 27.7
            dz = (self.z_range[0] - self.z_range[1] )/ (self.z_grid_points - 1) 
            SQUARE_ARC_MIN_TO_SQUARE_DEG = 3600
            zgrid = np.linspace(self.z_range[0], self.z_range[0], self.z_grid_points)
            nz_fit =  (zgrid**2)*np.exp(-1*(zgrid/0.28)*(0.90))
            NORM = (NEFF) / np.sum(nz_fit)
            nz_fit = nz_fit * NORM * SQUARE_ARC_MIN_TO_SQUARE_DEG 
            zedges = np.append(zgrid, zgrid[-1] + dz ) - dz/2



        values, edges = np.histogram(self.sim_cat['redshift_true'], bins=zedges)
        values_sim = values/self.sim_area
        z_frac = nz_fit / values_sim
        z_frac = np.minimum(z_frac, np.ones(len(z_frac))*0.99)
        
        self.z_frac = z_frac
        self.z_grid = z_grid
        self.z_edges = z_edges
        self.nz_fit = nz_fit

        # save the new z center
        path_z_grid = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/selection_z_centers/shear/{self.z_grid_points}_point_zgrid.npy'
        np.save(path_z_grid, self.z_grid)


    def generate_threshold(self):
        
        thres_list = []
        
        for i in range(len(self.z_grid)):
            
            this_zmin = self.z_edges[i]
            this_zmax = self.z_edges[i + 1]
            this_cat = self.sim_cat[np.logical_and(self.sim_cat['redshift_true']>this_zmin, self.sim_cat['redshift_true']<this_zmax)]
        
            if len(this_cat) == 0:
                
                print(f"Empty bin: zmin={this_zmin}, zmax={this_zmax}")
                this_thres = 10**40 # set threshold to high value to not select anything

            else:
                this_thres = np.percentile(a = this_cat[f'{self.threshold_col}'], q = 100-self.z_frac[i]*100)
    

            thres_list.append(this_thres)
    
        self.thres_list = thres_list

    def produce_shear_mock(self):
  
        thres_of_z = interpolate.interp1d(self.z_grid, self.thres_list,  fill_value="extrapolate", bounds_error=False)
        threshold_all = thres_of_z(self.sim_cat['redshift_true'])
    
        # save the threshold values
        path_threshold = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/selection_thresholds/shear/{self.threshold_col}_thres.npy'
        np.save(path_threshold, threshold_all)
    
        mask_abundance = self.sim_cat[self.threshold_col] > threshold_all
        mock_cat = self.sim_cat[mask_abundance]
    
        return mock_cat

        

        