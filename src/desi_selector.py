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


class DesiSelector:

    def __init__(self, 
                 desi_tracer,
                 path_desi_tracer,
                 path_sim,
                 model_calibration,
                 sim_patches,
                 sim_area=1121,
                 z_range = [0,2], 
                 z_grid_points=481
                 ):

        self.desi_tracer = desi_tracer
        self.path_desi_tracer = path_desi_tracer
        self.path_sim = path_sim
        self.model_calibration = model_calibration
        self.sim_patches = sim_patches
        self.sim_area = sim_area
        self.z_range = z_range
        self.z_grid_points = z_grid_points

        # column to use for threshold calculation
        if self.desi_tracer == 'bgs':
            self.threshold_col = 'lsst_r'

        elif self.desi_tracer == 'lrg':
            self.threshold_col = 'log_halo_mass'

        elif self.desi_tracer == 'elg':
            self.threshold_col = 'log_sfr'

        elif self.desi_tracer == 'qso':
            self.threshold_col = 'black_hole_mass'
    
    
    # def load_sim_cat(self):
        
        dict_model_calibrations = {'tng': 'tng_2025_11_06', 
                           'um': 'smdpl_dr1_2025_11_07',
                           'gal': 'galacticus_in_plus_ex_situ_2025_11_10',
                           'hlwas_cosmos': 'hlwas_cosmos_260120_UM_2026_01_22',
                           'cosmos': 'cosmos_260120_UM_2026_01_21'}
        
        path_sim_data = Path(f"{self.path_sim}/{dict_model_calibrations[self.model_calibration]}")
        list_sim_data = list(path_sim_data.glob("*.hdf5"))
        
        dataset = oc.open(list_sim_data)

        if self.desi_tracer == 'bgs':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_r']

        elif self.desi_tracer == 'lrg':
            columns = ['ra', 'dec', 'redshift_true', 'logmp_obs']

        elif self.desi_tracer == 'elg':
            columns = ['ra', 'dec', 'redshift_true', 'logsm_obs', 'logssfr_obs', 'lsst_g', 'lsst_r', 'lsst_z']

        elif self.desi_tracer == 'qso':
            columns = ['ra', 'dec', 'redshift_true', 'black_hole_mass', 'black_hole_eddington_ratio']
        
        
        dataset = dataset.select(columns)
        dataset = dataset.with_redshift_range(self.z_range[0], self.z_range[1])
        sim_cat = dataset.data.to_pandas()

        if self.desi_tracer == 'lrg':
            sim_cat.rename(columns={'logmp_obs': 'log_halo_mass'}, inplace=True)

        if self.desi_tracer == 'elg':
            sim_cat['log_sfr'] = sim_cat['logsm_obs'] + sim_cat['logssfr_obs']

        self.sim_cat = sim_cat

    
    def rebin_desi_tracer(self):
        
        # load the desi tracer data we are abundance matching to and get bin edges and centers

        if self.desi_tracer == 'bgs':
            tracer_data = np.loadtxt(self.path_desi_tracer)
            z_bin_center = tracer_data[:,0]
            z_bin_min = tracer_data[:,1]
            z_bin_max = tracer_data[:,2]

        
        elif self.desi_tracer == 'lrg':
            tracer_data = pd.read_csv(self.path_desi_tracer, index_col=False)
            z_bin_min = tracer_data['zmin']
            z_bin_max = tracer_data['zmax']
            z_bin_center = (z_bin_min + z_bin_max) / 2

        
        elif self.desi_tracer == 'elg':
            tracer_data = table.Table.read(self.path_desi_tracer,  format='ascii.ecsv')
            z_bin_min = tracer_data['ZMIN']
            z_bin_max = tracer_data['ZMAX']
            z_bin_center = (z_bin_min + z_bin_max) / 2 

        
        elif self.desi_tracer == 'qso':
            tracer_data = table.Table.read(self.path_desi_tracer,  format='ascii.ecsv')
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            z_bin_center = tracer_data['z'][z_mask]
            z_bin_min = z_bin_center - 0.050/2
            z_bin_max = z_bin_center + 0.050/2
        

        
        # get the desi tracer number/deg2 data we want to match to 
        if self.desi_tracer == 'bgs':
            EFF_AREA_NORTH = 5108.04
            EFF_AREA_SOUTH = 2071.91
            
            path_north = '/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/desi_sv_data/desi_bgs_ts_zenodo/BGS_BRIGHT_NGC_nz.txt'
            path_south = '/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/desi_sv_data/desi_bgs_ts_zenodo/BGS_BRIGHT_SGC_nz.txt'
            data_north = np.loadtxt(path_north)
            data_south = np.loadtxt(path_south)
            
            n_bin_north = data_north[:,4]
            n_bin_south = data_south[:,4]
            nz_avg = ((n_bin_north / EFF_AREA_NORTH) + (n_bin_south / EFF_AREA_SOUTH)) / 2 

        
        elif self.desi_tracer == 'lrg':
            nz_avg = tracer_data['n_desi_lrg']

        
        elif self.desi_tracer == 'elg':
            LOP_NORTH_AREA = 4400
            LOP_SOUTH_DECAL_AREA = 8500
            LOP_SOUTH_DES_AREA = 1100
            TOTAL_DESI_AREA = 14000
            lop_north = tracer_data['ELG_LOP_NORTH']
            lop_south_decal = tracer_data['ELG_LOP_SOUTH_DECALS']
            lop_south_des = tracer_data['ELG_LOP_SOUTH_DES']
            nz_avg = (lop_north * LOP_NORTH_AREA + lop_south_decal * LOP_SOUTH_DECAL_AREA  + lop_south_des * LOP_SOUTH_DES_AREA )/(TOTAL_DESI_AREA)


        elif self.desi_tracer == 'qso':
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            nz_north = tracer_data['n_z_north'][z_mask]
            nz_south = tracer_data['n_z_south'][z_mask]
            nz_avg = (nz_north + nz_south) / 2 
        

        repeat_n = int((self.z_grid_points-1)/len(z_bin_center))
        new_z_bin_min = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[:-1]
        new_z_bin_max = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[1:]
        new_z_center = (new_z_bin_max + new_z_bin_min) / 2

        interp_func = interp1d(z_bin_center, nz_avg, fill_value=0, bounds_error=False)
        interp_nz_avg = interp_func(new_z_center) / repeat_n

        zgrid = np.linspace(np.min(z_bin_min), np.max(z_bin_max), self.z_grid_points)
        values, edges = np.histogram(self.sim_cat['redshift_true'], bins=zgrid)
        values_sim = values / self.sim_area
        z_frac = interp_nz_avg / values_sim
        z_frac = np.minimum(z_frac, np.ones(len(z_frac))*0.99)

        self.new_z_bin_min = new_z_bin_min
        self.new_z_center = new_z_center
        self.new_z_bin_max = new_z_bin_max
        self.z_frac = z_frac
        self.nz_avg = nz_avg
        
            #return (new_z_bin_min, new_z_center, new_z_bin_max, z_frac)
        
    
    def generate_threshold(self):
        
        thres_list = []
        for i in range(len(self.new_z_center)):
            
            this_zmin = self.new_z_bin_min[i]
            this_zmax = self.new_z_bin_max[i]
            this_cat = self.sim_cat[np.logical_and(self.sim_cat['redshift_true']>this_zmin, self.sim_cat['redshift_true']<this_zmax)]
        
            if len(this_cat) == 0:
                print(f"Empty bin: zmin={this_zmin}, zmax={this_zmax}")
            
            this_thres = np.percentile(a = this_cat[self.threshold_col], q = 100-self.z_frac[i]*100)
            thres_list.append(this_thres)
                        
                     
        self.thres_list = thres_list

    def produce_desi_mock(self):

        thres_of_z = interpolate.interp1d(self.new_z_center, self.thres_list,  fill_value=9E20, bounds_error=False)
        threshold_all = thres_of_z(self.sim_cat['redshift_true'])
        mask_abundance = self.sim_cat[self.threshold_col] > threshold_all
        desi_mock_cat = self.sim_cat[mask_abundance]

        return desi_mock_cat

    # def apply_color_cuts
    
    # def generate_mock_randoms
        
    # def measure_auto_corr(self):

    # def compare_auto_corr(self):
        


    # def run(self):

    #     self.prepare_threshold()

    #     self.generate_threshold()

    #     self.produce_mock()
        