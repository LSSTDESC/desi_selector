import opencosmo as oc
import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import table
from pathlib import Path
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from diffsky.experimental import lc_utils
from diffsky.data_loaders.hacc_utils import lightcone_utils
import jax.random as jran
import treecorr as tc
import healpy as hp


class DesiSelector:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 desi_tracer,
                 path_desi_tracer,
                 path_sim,
                 model_calibration,
                 sim_patches,
                 z_range = [0,2], 
                 z_grid_points=481
                 ):

        self.desi_tracer = desi_tracer
        self.path_desi_tracer = path_desi_tracer
        self.path_sim = path_sim
        self.model_calibration = model_calibration
        self.sim_patches = sim_patches
        # self.sim_area = sim_area
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
    
    
        
        dict_model_calibrations = {'tng': 'tng_latest', 
                           'um': 'smdpl_dr1_latest',
                           'gal': 'galacticus_in_plus_ex_situ_latest',
                           'hlwas_cosmos': 'hlwas_cosmos_260120_UM_latest',
                           'cosmos': 'cosmos_260120_UM_latest'}
        
        path_sim_data = Path(f"{self.path_sim}/{dict_model_calibrations[self.model_calibration]}")
        list_sim_data = list(f for f in path_sim_data.glob("*.hdf5") if f.stem.startswith("lc_cores"))

        # Calculate the total area the mocks span on the sky 
        dataset = oc.open(list_sim_data)
        pixels = dataset.region.pixels
        nside = dataset.region.nside
        sim_area = len(pixels)*hp.nside2pixarea(nside, degrees=True)
        self.sim_area=sim_area

        print(f'The total area spanned by these mocks is: {self.sim_area}'

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
        sim_cat['distance'] = DesiSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value

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
        z_frac = np.nan_to_num(z_frac, nan=0.0)
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
                this_thres = 10**50 # set threshold to high value to not select anything

            else:
                this_thres = np.percentile(a = this_cat[self.threshold_col], q = 100-self.z_frac[i]*100)

            thres_list.append(this_thres)
                        
                     
        self.thres_list = thres_list

    def produce_desi_mock(self):

        
        thres_of_z = interpolate.interp1d(self.new_z_center, self.thres_list,  fill_value=9E20, bounds_error=False)
        threshold_all = thres_of_z(self.sim_cat['redshift_true'])
        mask_abundance = self.sim_cat[self.threshold_col] > threshold_all
        mock_cat = self.sim_cat[mask_abundance]

        return mock_cat

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
    

    def measure_autocorr(self, mock_cat, rand_cat, z_range_clustering=[0.8, 1.1], min_sep=1, max_sep=200, nbins=100):
    

        
        mask_mock_z_cut = np.logical_and(mock_cat['redshift_true'] > z_range_clustering[0], mock_cat['redshift_true'] < z_range_clustering[1]) 
        mock_cat = mock_cat[mask_mock_z_cut]
    
        mask_rand_z_cut = np.logical_and(rand_cat['redshift_true'] > z_range_clustering[0], rand_cat['redshift_true'] < z_range_clustering[1]) 
        rand_cat = rand_cat[mask_rand_z_cut]
    
        ra = mock_cat['ra']
        dec = mock_cat['dec']
        s = mock_cat['distance']
        
        ra_rand = rand_cat['ra']
        dec_rand = rand_cat['dec']
        s_rand = rand_cat['distance']
    
        print('The ratio of data to randoms is:', len(ra_rand)/ len(ra))
    
        # calculate the correlation using s
    
        dg_to_r = np.pi / 180
        
        tc_cat_s = tc.Catalog(ra=ra*dg_to_r, dec=dec*dg_to_r, r=s, ra_units='radians', dec_units='radians')
        tc_rnd_s = tc.Catalog(ra=ra_rand*dg_to_r, dec=dec_rand*dg_to_r, r=s_rand, ra_units='radians', dec_units='radians')
        
        
        dd_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        dr_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rr_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        
        dd_s.process(tc_cat_s, metric='Euclidean')
        dr_s.process(tc_cat_s, tc_rnd_s, metric='Euclidean')
        rr_s.process(tc_rnd_s, metric='Euclidean')
        xi_s, var_xi = dd_s.calculateXi(rr=rr_s, dr=dr_s)
        xi_s_naive, var_xi_naive = dd_s.calculateXi(rr=rr_s)
        sep = DesiSelector.cosmo.h * np.exp(dd_s.meanlogr)
    
        return {'sep': sep,
               'xi': xi_s,
                'var_xi': var_xi,
               'dd': dd_s,
               'dr': dr_s,
               "rr": rr_s,
               "xi_naive": xi_s_naive,
               'var_xi_naive': var_xi_naive}
    
    
    # def measure_auto_corr(self):

    # def compare_auto_corr(self):
        


    # def run(self):

    #     self.prepare_threshold()

    #     self.generate_threshold()

    #     self.produce_mock()
        