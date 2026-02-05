import opencosmo as oc
import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import table
import h5py
from pathlib import Path
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from diffsky.experimental import lc_utils
from diffsky.data_loaders.hacc_utils import lightcone_utils
import jax.random as jran


class DesiPhotSelector:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 desi_tracer,
                 path_sim,
                 model_calibration,
                 sim_patches,
                 sim_area=1121
                 ):

        self.desi_tracer = desi_tracer
        self.path_sim = path_sim
        self.model_calibration = model_calibration
        self.sim_patches = sim_patches
        self.sim_area = sim_area

    

        dict_model_calibrations = {'tng': 'tng_latest', 
                           'um': 'smdpl_dr1_latest',
                           'gal': 'galacticus_in_plus_ex_situ_latest',
                           'hlwas_cosmos': 'hlwas_cosmos_260120_UM_latest',
                           'cosmos': 'cosmos_260120_UM_latest'}
        
        path_sim_data = Path(f"{self.path_sim}/{dict_model_calibrations[self.model_calibration]}")
        list_sim_data = list(f for f in path_sim_data.glob("*.hdf5") if f.stem.startswith("lc_cores"))
        dataset = oc.open(list_sim_data)

        if self.desi_tracer == 'bgs':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_r']

        elif self.desi_tracer == 'lrg':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_g', 'lsst_r', 'lsst_z']

        elif self.desi_tracer == 'elg':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_g', 'lsst_r', 'lsst_z']

        elif self.desi_tracer == 'qso':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_g', 'lsst_r', 'lsst_z']
        
        
        dataset = dataset.select(columns)
        dataset = dataset.with_redshift_range(self.z_range[0], self.z_range[1])
        sim_cat = dataset.data.to_pandas()
        sim_cat['distance'] = DesiPhotSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value
        self.sim_cat = sim_cat


    def produce_desi_mock(self):

        
        if self.desi_tracer == 'bgs':
            
            R_MAG_LIMIT = 19.5
            mask_photometry = self.sim_cat['lsst_r'] <  R_MAG_LIMIT
        
        elif self.desi_tracer == 'elg':
            
            COLOR_CUT1_SLOPE = 0.50
            COLOR_CUT1_INTERCEPT = 0.1
            COLOR_CUT2_SLOPE = -1.20
            COLOR_CUT2_INTERCEPT = 1.3
            G_FIBER_LIMIT = 24.1 
            G_FIBER_MAG_OFFSET = 0.65
            G_MAG_LIMIT = G_FIBER_LIMIT - G_FIBER_MAG_OFFSET # Based on average difference between fiber and mag 
            mask_color = np.logical_and((self.sim_cat['lsst_g'] - self.sim_cat['lsst_r']) < COLOR_CUT1_SLOPE*(self.sim_cat['lsst_r'] - self.sim_cat['lsst_z']) + COLOR_CUT1_INTERCEPT, 
                                            (self.sim_cat['lsst_g'] - self.sim_cat['lsst_r']) < COLOR_CUT2_SLOPE*(self.sim_cat['lsst_r'] - self.sim_cat['lsst_z']) + COLOR_CUT2_INTERCEPT)
            mask_g_mag = self.sim_cat['lsst_g'] < G_MAG_LIMIT
            mask_photometry = np.logical_and(mask_color, mask_g_mag)

        return mock_cat[mask_photometry]

    
    def produce_desi_rands(self, mock_cat=None):


        RAND_TO_DATA_RATIO = 10
        npatches = len(self.sim_patches)
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
    
        
        list_ra = []
        list_dec = []
        
        for patch in self.sim_patches:
            
            ra_loop, dec_loop = lc_utils.mc_lightcone_random_ra_dec(ran_key=ran_key, npts=ntot, ra_min=ra_min[patch],
            ra_max=ra_max[patch], dec_min=dec_min[patch], dec_max=dec_max[patch])
    
            list_ra.append(ra_loop)
            list_dec.append(dec_loop)
                                        
            
        rand_ra = np.concatenate(list_ra)
        rand_dec = np.concatenate(list_dec)
            
        list_rand_cols = np.column_stack([rand_ra, rand_dec])
        rand_cat = pd.DataFrame(list_rand_cols, columns=['ra', 'dec'])
        rand_cat = rand_cat.reset_index(drop=True) 
        mock_cat_temp = mock_cat.reset_index(drop=True).sample(len(rand_cat), replace=True)
        rand_cat['distance'] = mock_cat_temp['distance'].to_numpy()
        rand_cat['redshift_true'] = mock_cat_temp['redshift_true'].to_numpy()
    
        return rand_cat
    
        
    # def measure_auto_corr(self):

    # def compare_auto_corr(self):
        


    # def run(self):

    #     self.prepare_threshold()

    #     self.generate_threshold()

    #     self.produce_mock()
        