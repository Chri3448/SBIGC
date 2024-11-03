import shutil
import os
import pdb
import yaml
import torch   
import copy
import numpy as np
import scipy as sp
import healpy as hp
from pathlib import Path
import scipy.interpolate as interp
import astropy.units as units
from astropy.io import fits
from sbi.utils import process_prior
from .utils import get_loader, setup_cosmology, differential_comoving_volume_interpolator, recursive_dictionary_check
from .sources.extragalactic_point_sources import *

'''
The class is used to generate mock datasets, and to interface that generation with ELFI sampling methods.
'''

class LFI_unbinned:

    def __init__(self, yaml_file, default_yaml=Path(__file__).parent / 'default_values.yaml'):
       # set output width for output
        self._terminal_columns = shutil.get_terminal_size().columns

        # load the user config file and default config file
        loader = get_loader()
        with open(yaml_file, 'r') as infile:
            config_dict = yaml.load(infile, Loader=loader)
        with open(default_yaml, 'r') as infile:
            default_dict = yaml.load(infile, Loader=loader)

        # read the required parameters
        required_params = ['job_name', 'output_path', 'sources', 'priors']
        for required_param in required_params:
            try:
                setattr(self, required_param, config_dict.pop(required_param))
            # raise an exception if the user has not provided all of the required parameters
            except KeyError:
                raise KeyError(f"You must provide {required_param} for the analysis.")

        # parse the user config and the default config
        updated_attributes_config = self._parse_yaml_dict(config_dict)
        updated_atrributes_default = self._parse_yaml_dict(default_dict)

        # begin initializing
        self.print_output('Initializing a likelihood free inference object for extragalactic gamma ray simulation', kind='verbose', header=2, footer=2)
        self.print_output(f'Custom settings will be read from {yaml_file}', kind='verbose')
        self.print_output(f'Default settings will be read from {default_yaml}', kind='verbose')

        
        self.print_output('Settings read successfully!', kind='verbose')
        self.print_output(f'Settings read from config file: {required_params + updated_attributes_config}', kind='debug', prepend=True)
        self.print_output(f'Settings read from default file: {updated_atrributes_default}', kind='debug', prepend=True)

        # create working directory
        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            print(f'Warning: {self.output_path} already exists! Overwriting...')


        # check that exposure has been provided. if it hasn't been, set the exposure using other provided parameters
        self._set_exposure()

        # setup cosmology and cosmological functions
        self.cosmo = setup_cosmology()

        # assign a number density function to each source class
        # self._ensure_number_density_functions()

        # get priors for all sampled parameters and set params to parse samples
        self._parse_parameter_samples_and_priors()

        # print out summary of initialization
        self.print_output(f'Summary of source settings for this analysis', kind='verbose', header=2, footer=1)
        # we don't want to print out any functions that are stored as dictionary values
        new_sources = copy.deepcopy(self.sources)

        # need to replace functions with a description of the function that is being used in the sources dict
        new_sources = recursive_dictionary_check(new_sources)

        self.print_output(yaml.dump(new_sources, Dumper=yaml.Dumper, sort_keys=False), kind='verbose')
        self.print_output('Initialization complete!', kind='verbose', header=1, footer=1)
        self.print_output('', kind='verbose')


    #####################
    ##### PARSEYAML #####
    #####################
    def _parse_yaml_dict(self, yaml_dict: dict) -> list[str]:
        """Read in attributes from yaml file.

        Used to set attributes from the provided config file that are not mandatory and to set any attributes to their default value that are not user provided.

        :param yaml_dict: input dict from yaml file containing settings for analysis, source information, and priors
        :type yaml_dict: dict

        :returns: all attributes that are updated during the function call
        :rtype: list[str]
        """
        # keep track of which attributes are created/updated
        updated_attributes = []

        # loop through settings dict
        for key in yaml_dict: 
            # loop through the analysis settings block and set attributes
            if key == 'analysis_settings':
                for setting_name, setting in yaml_dict[key].items():
                    # only set the attribute if it doesn't currently exist
                    if not hasattr(self, setting_name):
                        setattr(self, setting_name, setting)
                        updated_attributes.append(key+':'+setting_name)
            # loop through the sources and set all the unset default source parameters
            # this is a required key and thus removed for the config dict
            # so only applies to default dict
            elif key == 'sources':
                # loop through each source
                for source_name, source in self.sources.items():
                    # check that a source class has been provided
                    if not 'source_class' in source:
                        if source_name in yaml_dict[key]:
                            source['source_class'] = source_name
                        else:
                            raise NotImplementedError(f'Problem parsing source {source_name}. Specify a "source_class" in the {source_name} settings block or use one of the default names: {yaml_dict[key].keys()}')
                    default_vals = yaml_dict[key][source['source_class']]
                    for default_param_name, default_param in default_vals.items():
                        if not default_param_name in source:
                            source[default_param_name] = default_param
                            updated_attributes.append(key+':'+source_name+':'+default_param_name)

                    self.sources[source_name] = eval(source['source_class'])(self, source)
                    print(self.sources[source_name])

            elif key in ['obs_info', 'summary_properties']:
                if not hasattr(self, key):
                    setattr(self, key, {})
                for setting_name, setting in yaml_dict[key].items():
                    # only set the attribute if it doesn't currently exist
                    if not setting_name in getattr(self, key):
                        getattr(self, key)[setting_name] = setting

                        updated_attributes.append(key+':'+setting_name)
            # this is removed for the config dict so only applies to the default dict
            elif key == 'priors':
                pass
            # any other parameter is set
            else:
                if not hasattr(self, key):
                    setattr(self, key, yaml_dict[key])
                    updated_attributes.append(key)

        return updated_attributes

    def _set_exposure(self):
        self.print_output(f'Setting exposure', kind='verbose', header=1)

        s_to_yr_constant = 3600 * 24 * 7 * 52
        # check for exposure map file
        if 'exposure_map_file' in self.obs_info:
            # read exposure map from file
            exposure_map = np.load(self.obs_info['exposure_map_file'])
            if self.obs_info['exposure_units'] == 'cm2 yr':
                exposure_map *= s_to_yr_constant
            # match self.obs_info['exposure_units']:
            #     case 'cm2 yr':
            #         exposure_map *= s_to_yr_constant
            #     case 'cm2 s':
            #         pass
            #     case other:
                    # raise NotImplementedError(f'Unrecognized exposure units {other}. Try "cm2 yr" or "cm2 s".')

            nside = hp.npix2nside(len(exposure_map))
            self.obs_info['exposure_map'] = lambda theta, phi: exposure_map[hp.ang2pix(nside, theta, phi)]
            self.exposure = np.amax(exposure_map)
            self.print_output("Loading exposure map from {self.obs_info['exposure_map_file']}", kind='verbose')
            self.print_output("Exposure set to {self.exposure} cm2 s", kind='verbose')

        if not hasattr(self, 'exposure'):
            try:
                setattr(self, 'exposure', self.obs_info['area'] * self.obs_info['obs_time'] * self.obs_info['fov_factor'])
                if self.obs_info['exposure_units'] == 'cm2 yr':
                    self.exposure *= s_to_yr_constant
                self.print_output(f"Exposure set using default values (units assumed to be [area]=cm2, [obs_time]=yr) to {self.exposure} cm2 s. If any default values were overwritten, double-check units.", kind='verbose')
            except AttributeError:
                raise AttributeError("Exposure is not specified. Either provide an exposure or the area, observation time and fov factor.")

    def _parse_parameter_samples_and_priors(self):
        """Assign indices of the sample array for each source.

        These indices are used to slice up the parameter sample array for into the appropriate chunks for all the sources.
        """
        i = 0
        priors = []
        for source_name, source in self.sources.items():
            self.print_output(f'Loading priors for {source_name}', kind='verbose', header=1)
            if not source_name in self.priors:
                raise AttributeError(f'Priors required for {source_name}.')

            source.fixed_params = {}
            sampled_parameter_names = []
            fixed_parameter_names = []
            for prior_name, prior in self.priors[source_name].items():
                if prior['type'] != 'fixed':
                    sampled_parameter_names.append(prior_name)
                    prior_func = self._get_prior_func(prior)
                    processed_prior, *_ = process_prior(prior_func)
                    priors.append(processed_prior)
                else:
                    # fixed parameters will not be changed
                    fixed_parameter_names.append(prior_name)
                    prior_func = f"Fixed({prior['value']}"
                    # source['fixed_params'].update({prior_name: prior['value']})
                    source.fixed_params[prior_name] =  prior['value']

                self.print_output(f'Prior loaded for {prior_name}: {prior_func}', kind='debug', prepend=True)

            # set params to slice the sample for each source class
            source.sample_index_min = i
            i += len(sampled_parameter_names)
            source.sample_index_max = i

            # Check that we have all the necessary priors
            param_names = source.get_param_names()
            same_set = set(param_names).difference(set(sampled_parameter_names + fixed_parameter_names))

            if len(same_set) != 0:
                raise AttributeError(f'Missing prior(s) for {same_set}')

            # Set the names of the sampled parameters to pass to num density function
            source.sample_slice_names = sampled_parameter_names

            self.print_output(f'{source_name} loaded with {len(sampled_parameter_names)} parameters', kind='verbose')

        # Set attributes
        self.number_of_sampled_parameters = i
        self.priors_info = self.priors
        self.priors = priors

        self.print_output(f'Total number of sampled parameters is: {i}.', kind='verbose')

    def _get_prior_func(self, prior):
        if prior['type'].lower() in ['normal', 'norm', 'gaussian', 'gauss']:
            return self.prior_normal(prior['mu'], prior['sigma'])
        elif prior['type'].lower() in ['uniform', 'unif', 'box']:
            return self.prior_uniform(prior['min'], prior['max'])
        else:
            raise NotImplementedError(f'Unrecognized or unimplemented prior type {prior["type"]}')

         
    ##########################################################################
    '''
    Basic statistical functions

    '''
    ##########################################################################
    def draw_from_pdf(self, cc, Pc, Ndraws):
        # draw random counts from P(c)
        cdf = np.cumsum(Pc)
        rands = np.random.rand(Ndraws)
        # Draw Ndraws times from Pc
        d_vec = np.searchsorted(cdf, rands)
        return d_vec
    
    #Takes a 2D array. pdf[x,y] should equal pdf(x,y). Returns two 1D arrays of x and y indices.
    #pdf*dx*dy must be passed to this func as pdf for normalization. If x or y are not linspaced, the pdf dimensions should be (n-1,m-1),
    #and the last indices of x and y will not be drawn
    def draw_from_2D_pdf(self, pdf, Ndraws = 0):
        if Ndraws == 0:
            Ndraws = int(round(np.sum(pdf)))
        flipped = False
        if np.shape(pdf)[0] > np.shape(pdf)[1]:
            pdf = pdf.T
            flipped = True
        self.print_output(f'Drawing {Ndraws} sources', kind='verbose')
        x_pdf = np.sum(pdf, axis = 1)/np.sum(pdf)
        x_cdf = np.cumsum(x_pdf)
        x_rands = np.random.rand(Ndraws)
        x_indices = np.searchsorted(x_cdf, x_rands)
        y_cdfs = np.cumsum(pdf, axis = 1)/np.tile(np.sum(pdf, axis = 1), (np.size(pdf[0,:]),1)).T
        y_rands = np.random.rand(Ndraws)
        y_indices = np.zeros(np.size(x_indices), dtype = 'int')
        for i in range(np.size(pdf[:,0])):
            source_positions = np.where(x_indices == i)
            y_indices[source_positions] = np.searchsorted(y_cdfs[i,:], y_rands[source_positions])
        if flipped:
            return y_indices, x_indices
        else:
            return x_indices, y_indices

    def draw_from_isotropic_EPDF(self, params, source_index, exposure, Sangle, npix):
        '''
        Given a binned spectrum, draw photon counts in energy pixels for npix pixels

        Output will have dimensions (N_pix, N_energy_bins)
        '''
        if (not self.is_istropic_list[source_index]):
            print("attempting to draw from isotropic source that is not isotropic!!!")
        
        output = np.zeros((npix, self.nEbins))

        #Compute total flux in each energy bin (THIS IS INTEGRATED OVER ENERGY)
        E_flux = np.zeros(self.nEbins)
        for ei in range(0,self.nEbins):
            e = np.geomspace(self.Ebins[ei], self.Ebins[ei+1], 200)
            de = e[1:] - e[:-1]
            dnde = self.PDF_spec[source_index][1](params, e, self.args)
            E_flux[ei] = np.sum(0.5*de*(dnde[1:] + dnde[:-1]))

        #In principle, this could handle non-istropic PDFs very easily
        if (self.is_poisson_list[source_index]):
            #if poisson, then compute total flux
            dE_bins = self.Ebins[1:] - self.Ebins[:-1]
            total_flux = np.sum(E_flux)
            # draw total photon count for each pixel
            mean_photon_count_per_pix = exposure*Sangle*total_flux
            photon_counts_per_pix = np.random.poisson(lam = mean_photon_count_per_pix, size = npix)
        else:
            Pc = self.PDF_spec[source_index][0](params, self.Cbins, self.args)
            photon_counts_per_pix = self.draw_from_pdf(self.Cbins, Pc, npix)

        if (self.nEbins == 1):
            output[:,0] = photon_counts_per_pix
        if (self.nEbins > 1):
            #Set up spectral weights - these are the fraction of photons that fall into each energy bin            
            spec_weights = E_flux/np.sum(E_flux)
            
            #draw np.sum(photon counts per pix) times from a PDF
            #this is energy bin of all photons
            ebin_arr = self.draw_from_pdf(np.arange(len(spec_weights)), spec_weights, np.sum(photon_counts_per_pix))
                                          
            pix_indices = np.repeat(np.arange(npix),photon_counts_per_pix)
            for ei in range(0,self.nEbins):
                in_bin = np.where(ebin_arr == ei)[0]
                output[:,ei] = np.bincount(pix_indices[in_bin], minlength = npix)
                
        return output.astype('int')
                
    #####################
    ####### PRIOR #######
    #####################
    def prior_normal(self, mu: float, sigma: float):
        """Normal prior.

        :param mu: mean of distribution (also called loc)
        :type mu: float
        :param sigma: standard deviation of distribution (also called scale)
        :type sigma: float
        """
        return torch.distributions.normal.Normal(torch.tensor([float(mu)]), torch.tensor([float(sigma)]))

    def prior_uniform(self, umin: float, umax: float):
        """Uniform prior.

        :param umin: minimum value of distribution
        :type umin: float
        :param umax: maximum value of distribution
        :type umax: float
        """
        return torch.distributions.uniform.Uniform(torch.tensor([float(umin)]), torch.tensor([float(umax)]))


    ##########################################################################
    '''
    Data generation functions

    '''
    ##########################################################################
    
    
    def generate_mock_data_unbinned(self, input_params):
        '''
        The main mock data generation function
        '''
        #source_info = self.create_sources(input_params)
        #photon_info = self.generate_photons_from_sources(input_params, source_info)

        #Pseudocode
        '''
        loop over sources
           if (source is Poisson):
               #every poisson source should have a saved map corresponding to spatial variation
               multiply this map by energy-dependent exposure matrix

               draw photons from this map (how?)

           if (source is not Poisson):

               loop over radial bins
                    loop over luminosity bins
    

        '''
        pass


    def create_sources(self, input_params, grains = 1000):
        '''
        This function creates a list of sources, where each source has a radial distance, mass, and luminosity

        dndM = number density of sources in infinitessimal mass bin
        input and output masses are in units of the solar mass
        maxr = max radial distance from galactic center in kpc
        '''
        pass
    

    def generate_photons_from_sources(self, input_params, source_info):
        '''
        Function returns list of photon energies and sky positions
        '''
        pass        
    
    def draw_from_isotropic_background_unbinned(self, Ebins, exposure, Sangle):
        e, dnde = self.e_isotropic, self.dnde_isotropic
        f = interp.interp1d(e, dnde, kind='linear', fill_value=0.)
        #lowE = np.exp(np.log(e[0]) - (np.log(e[1]) - np.log(e[0]))/2)
        #highE = e[-1] + (e[-1] - e[-2])/2
        lowE = Ebins[0:-1]
        highE = Ebins[1:]
        e = np.geomspace(lowE, highE, 1000)
        dnde = f(e)
        int_terms = dnde[1:]*(e[1:]-e[:-1])
        num_photons = int(round(exposure*Sangle*np.sum(int_terms)))
        e_indices = self.draw_from_pdf(np.arange(0,len(e)-1), int_terms/np.sum(int_terms), num_photons)
        energies = e[e_indices]
        return energies
    
    #for non-isotropic healpix maps
    def draw_angles_and_energies(self, map_all, map_E, N_draws = 0):
        N_pix = map_all.shape[1]
        N_side = hp.npix2nside(N_pix)
        masked_i = np.where(np.abs(hp.pix2ang(N_side, np.linspace(0, N_pix-1, N_pix).astype('int'))[0] - np.pi/2) < np.radians(self.lat_cut))[0]
        new_map_all = np.copy(map_all)
        new_map_all[:, masked_i] *= 0
        dE = map_E[1:] - map_E[:-1]
        energy_indices, pixels = self.draw_from_2D_pdf(new_map_all[:-1,:]*self.exposure*(units.kpc.to('cm')**2)*(4*np.pi/N_pix)*(np.tile(dE, (N_pix,1)).T), N_draws)
        angles = hp.pix2ang(N_side, pixels)
        return np.array(angles).T, map_E[energy_indices]
    
    ##########################################################################
    '''
    Observational functions
    '''
    ##########################################################################
    

    def apply_PSF(self, photon_info, obs_info, energy_dependent = True):
        '''
        Applies energy dependent Fermi PSF assuming normal incidence
        If input energy is outside (10^0.75, 10^6.5) MeV, the PSF of the nearest energy bin is applied
        Only valid for Fermi pass 8
        '''
        psf_fits_path, event_type = obs_info['psf_fits_path'], obs_info['event_type']
        if not psf_fits_path.endswith(event_type[:-1] + '.fits'):
            print('!!!!WARNING!!!!\n event_type not found in given psf_fits file\n PSF not applied\n!!!!WARNING!!!!')
            return photon_info
        
        num_photons = np.size(photon_info['energies'])
        
        hdul = fits.open(psf_fits_path)
        scale_hdu = 'PSF_SCALING_PARAMS_' + event_type
        fit_hdu = 'RPSF_' + event_type
        C = hdul[scale_hdu].data[0][0][:-1]
        beta = -hdul[scale_hdu].data[0][0][2]
        fit_ebins = np.linspace(0.75, 6.5, 24)
        distances = np.zeros(num_photons)
        if not energy_dependent:
            average_i = np.seatchsorted(fit_ebins, np.log10(np.mean(photon_info['energies'])))
        #loop over energy bins in which params are defined
        for index in range(23):
            if index == 0:
                ebin_i = np.where(np.log10(photon_info['energies'])<fit_ebins[index+1])
            elif index == 22:
                ebin_i = np.where(np.log10(photon_info['energies'])>=fit_ebins[index])
            else:
                ebin_i = np.where(np.logical_and(np.log10(photon_info['energies'])>=fit_ebins[index], np.log10(photon_info['energies'])<fit_ebins[index+1]))
            if not energy_dependent:
                if index == average_i:
                    ebin_i = np.where(photon_info['energies']==photon_info['energies'])
                else:
                    ebin_i = np.where(photon_info['energies']!=photon_info['energies'])
            NTAIL = hdul[fit_hdu].data[0][5][7][index]
            SCORE = hdul[fit_hdu].data[0][6][7][index]
            STAIL = hdul[fit_hdu].data[0][7][7][index]
            GCORE = hdul[fit_hdu].data[0][8][7][index]
            GTAIL = hdul[fit_hdu].data[0][9][7][index]
            FCORE = 1/(1 + NTAIL*STAIL**2/SCORE**2)
            x_vals = 10**np.linspace(-1, 1.5, 1000)
            kingCORE = (1/(2*np.pi*SCORE**2))*(1-(1/GCORE))*(1+(1/(2*GCORE))*(x_vals**2/SCORE**2))**(-GCORE)
            kingTAIL = (1/(2*np.pi*STAIL**2))*(1-(1/GTAIL))*(1+(1/(2*GTAIL))*(x_vals**2/STAIL**2))**(-GTAIL)
            PSF = FCORE*kingCORE + (1-FCORE)*kingTAIL
            PDFx = 2*np.pi*x_vals[:-1]*PSF[:-1]*(x_vals[1:]-x_vals[:-1])
            x = x_vals[self.draw_from_pdf(x_vals[:-1], PDFx/np.sum(PDFx), np.size(ebin_i))]
            S_P = np.sqrt((C[0]*(photon_info['energies'][ebin_i]/100)**(-beta))**2 + C[1]**2)
            distances[ebin_i] = 2*np.sin(x*S_P/2)
        hdul.close()
        rotations = 2*np.pi*np.random.random(num_photons)
        #create orthonormal basis for each photon direction
        parallel = hp.ang2vec(photon_info['angles'][:,0], photon_info['angles'][:,1])
        perp1angles = photon_info['angles']
        perp1angles[:,0] += np.pi/2 
        over_indices = np.where(perp1angles[:,0]>np.pi)
        perp1angles[over_indices,0] = np.pi - perp1angles[over_indices,0]%np.pi
        perp1angles[over_indices,1] += np.pi
        perp1angles[over_indices,1] %= 2*np.pi
        perp1 = hp.ang2vec(perp1angles[:,0], perp1angles[:,1])
        perp2angles = photon_info['angles']
        perp2angles[:,0] = np.pi/2*np.ones(np.size(perp2angles[:,0]))
        perp2angles[:,1] += np.pi/2
        perp2angles[:,1] %= 2*np.pi
        perp2 = hp.ang2vec(perp2angles[:,0], perp2angles[:,1])
        #construct new direction from orthonormal basis
        new_parallel = np.tile(np.cos(distances), (3,1)).T*parallel
        new_perp = np.tile(np.sin(distances), (3,1)).T*(np.tile(np.cos(rotations), (3,1)).T*perp1 + np.tile(np.sin(rotations), (3,1)).T*perp2)
        photon_info['angles'] = np.array(hp.vec2ang(new_parallel+new_perp)).T
        '''
        delta_thetas = distances*np.cos(rotations)
        delta_phis = distances*np.sin(rotations)
        photon_info['angles'][:,0] += delta_thetas
        over_indices = np.where(np.logical_or(photon_info['angles'][:,0] > np.pi, photon_info['angles'][:,0] < 0))
        photon_info['angles'][over_indices,0] = np.pi - photon_info['angles'][over_indices,0]%np.pi
        photon_info['angles'][over_indices,1] += np.pi
        photon_info['angles'][:,1] += delta_phis
        photon_info['angles'][:,1] %= 2*np.pi
        '''
         
        return photon_info
    
    def apply_energy_dispersion(self, photon_info, obs_info):
        '''
        Applies Fermi energy dispersion assuming normal incidence
        If input energy is outside (10^0.75, 10^6.5) MeV, the energy dispersion of the nearest energy bin is applied
        Only valid for Fermi pass 8
        '''
        edisp_fits_path, event_type = obs_info['edisp_fits_path'], obs_info['event_type']
        if not edisp_fits_path.endswith(event_type[:-1] + '.fits'):
            print('!!!!WARNING!!!!\n event_type not found in given edisp_fits file\n Energy Dispersion not applied\n!!!!WARNING!!!!')
            return photon_info
        
        num_photons = np.size(photon_info['energies'])
        
        hdul = fits.open(edisp_fits_path)
        scale_hdu = 'EDISP_SCALING_PARAMS_' + event_type
        fit_hdu = 'ENERGY DISPERSION_' + event_type
        C = hdul[scale_hdu].data[0][0]
        fit_ebins = np.linspace(0.75, 6.5, 24)
        differences = np.zeros(num_photons)
        #loop over energy bins in which params are defined
        for index in range(23):
            if index == 0:
                ebin_i = np.where(np.log10(photon_info['energies'])<fit_ebins[index+1])
            if index == 22:
                ebin_i = np.where(np.log10(photon_info['energies'])>=fit_ebins[index])
            else:
                ebin_i = np.where(np.logical_and(np.log10(photon_info['energies'])>=fit_ebins[index], np.log10(photon_info['energies'])<fit_ebins[index+1]))
            F = hdul[fit_hdu].data[0][4][7][index]
            S1 = hdul[fit_hdu].data[0][5][7][index]
            K1 = hdul[fit_hdu].data[0][6][7][index]
            BIAS1 = hdul[fit_hdu].data[0][7][7][index]
            BIAS2 = hdul[fit_hdu].data[0][8][7][index]
            S2 = hdul[fit_hdu].data[0][9][7][index]
            K2 = hdul[fit_hdu].data[0][10][7][index]
            PINDEX1 = hdul[fit_hdu].data[0][11][7][index]
            PINDEX2 = hdul[fit_hdu].data[0][12][7][index]
            x_vals = np.linspace(-15, 15, 1000)
            x_low1, x_high1 = np.where(x_vals < BIAS1), np.where(x_vals >= BIAS1)
            x_low2, x_high2 = np.where(x_vals < BIAS2), np.where(x_vals >= BIAS2)
            g1, g2 = np.ones(1000), np.ones(1000)
            prefac1 = PINDEX1/(S1*sp.special.gamma(1/PINDEX1))*K1/(1+K1**2)
            prefac2 = PINDEX2/(S2*sp.special.gamma(1/PINDEX2))*K2/(1+K2**2)
            g1[x_low1] = prefac1*np.exp(-(1/(K1*S1)*np.abs(x_vals[x_low1]-BIAS1))**PINDEX1)
            g2[x_low2] = prefac2*np.exp(-(1/(K2*S2)*np.abs(x_vals[x_low2]-BIAS2))**PINDEX2)
            g1[x_high1] = prefac1*np.exp(-(K1/S1*np.abs(x_vals[x_high1]-BIAS1))**PINDEX1)
            g2[x_high2] = prefac2*np.exp(-(K2/S2*np.abs(x_vals[x_high2]-BIAS2))**PINDEX2)
            D = F*g1 + (1-F)*g2
            x = x_vals[self.draw_from_pdf(x_vals[:-1], D/np.sum(D), np.size(ebin_i))]
            E = photon_info['energies'][ebin_i]
            theta = 0
            S_D = C[0]*np.log10(E)**2 + C[1]*np.cos(theta)**2 + C[2]*np.log10(E) + C[3]*np.cos(theta) + C[4]*np.log10(E)*np.cos(theta) + C[5]
            differences[ebin_i] = x*E*S_D
        hdul.close()
        photon_info['energies'] += differences      
         
        return photon_info
    
    def apply_exposure(self, photon_info, obs_info):
        """Modify the generate photons to simulate a direction-dependent exposure.

        Photons are removed with probability 1 - exposure_map(theta, phi) / self.exposure. This assumes that photons have been generated with a max exposure of self.exposure and then are removed to simulate a directional dependence in the exposure map.

        The attribute self.exposure is a constant value. Photons are generated using this constant value.
        Optionally, an exposure_map can be provided that describes the dependence of exposure on the direction on the sky.
        
        'exposure_map' is a function that of the angles on the sky f(theta, phi) [in the same coordinate system as photon_info['angles'].
        If 'exposure_map' is a string, the function will try and load a healpix map and use it accordingly.

        :param photon_info: dictionary of photon_angles
        :param exposure_map: function of sky angles (theta, phi) describing the direction-dependent exposure. Or a file path name to a healpix map of the expsoure
        :returns: modified photon_dict
        """
        exposure_map = obs_info['exposure_map']
        
        # If there is no exposure map, we assume the exposure is constant and do not modify the photon list
        if exposure_map is None:
            return photon_info
        return photon_info

        # If exposure_map is a string, try to load a healpy map
        if isinstance(exposure_map, str): 
            try:
                hpx_map = np.load(exposure_map)
            except FileNotFoundError:
                raise FileNotFoundError('Exposure map must be either the relative file path name of a healpy exposure map or a function of sky angles f(theta,phi)')
            # Create the exposure_map funciton from the healpy exposure map
            nside = hp.npix2nside(len(hpx_map))
            def exposure_map(theta, phi): return hpx_map[hp.ang2pix(nside, theta, phi)]

        # Get the exposure for each photon
        exposures = exposure_map(*photon_info['angles'].T)

        # Double check that the exposure used to generate photons is consistent with this exposure map
        # If they are not consistent, the exposure_map is rescaled
        if not np.isclose(self.exposure, exposures.max()):
            exposures *= self.exposure / exposures.max()
            if self.verbose:
                print(f'\t-->Provided exposure and exposure_map are inconsistent.')
                print(f'\t-->Exposure map will be rescaled by a factor of {self.exposure / exposures.max()} to rectify')

        # Calculate the probability of rejecting the photon
        probabilities = 1 - exposures / self.exposure

        # Determine indices of removed photons randomly
        remove_photon_indices = (probabilities > np.random.random_sample(len(probabilities)))

        # take these photons out of the photon dict
        for key, values in photon_info.items():
            photon_info[key] = values[remove_photon_indices]

        if self.debug:
            print(f'\t-->Using direction-dependent exposure, {remove_photon_indices.sum()} photons removed')
            print(f'\t-->There are {~remove_photon_indices.sum()} photons remaining')

        return photon_info
    
    def mock_observe(self, photon_info, obs_info):
        #photon_info contains all information about individual photons
        #obs_info is a dictionary containing info about the observation process
        
        if np.any(np.isnan(photon_info['energies'])):
            return photon_info

        photon_info = self.apply_exposure(photon_info, obs_info)
        photon_info = self.apply_PSF(photon_info, obs_info)
        photon_info = self.apply_energy_dispersion(photon_info, obs_info)
        
        return photon_info
    
    ##########################################################################
    '''
    Summary statistic functions
    '''
    ##########################################################################

    def get_summary(self, photon_info, summary_properties = {'summary_type':'energy_dependent_histogram',
                                                             'map_type':'healpix',
                                                             'mask_galactic_plane': None,
                                                             'N_pix':12*64**2,
                                                             'mask_galactic_center_latitude':None, #in radians
                                                             'N_energy_bins':10,
                                                              'histogram_properties':{'Nbins':10, 'Cmax_hist': 10, 'Cmin_hist': 0, 'energy_bins_to_use':'all'}
                                                            }):

        if (summary_properties['summary_type'] == 'energy_dependent_histogram'):
            summary = self.get_energy_dependent_histogram(photon_info, summary_properties)
        if (summary_properties['summary_type'] == 'energy_dependent_map'):
            summary = self.get_energy_dependent_map(photon_info, summary_properties)
        
        return summary
    
    def get_energy_dependent_histogram(self, photon_info, summary_properties):
        # Calculate the energy-dependent histogram given
        
        if 'valid' in photon_info:
            if not photon_info['valid']:
                return np.zeros((1, summary_properties['N_energy_bins'] * summary_properties['histogram_properties']['Nbins'])) * np.nan

        emap = self.get_energy_dependent_map(photon_info, summary_properties)
        energy_dependent_histogram = self.get_energy_dependent_histogram_from_map(emap, summary_properties)
        
        return energy_dependent_histogram
    
    def get_energy_dependent_map(self, photon_info, summary_properties):
        '''
        Given unbinned photon data, return maps with dimension npix x N_energy

        map_type can be healpix or internal
        '''
        
        #The output map
        N_pix = summary_properties['N_pix']
        N_energy_bins = summary_properties['N_energy_bins']
        # TO DO: implement batch generation of photon info
        N_batch = 1 
        
        #For each batch, construct an object which is a map for each energy bin
        energy_dependent_map = np.zeros((N_batch, N_pix, N_energy_bins))

        map_type = summary_properties['map_type']
        if (map_type == 'healpix'):
            NSIDE = np.sqrt(N_pix/12).astype('int')
            pixels = hp.ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])
        elif (map_type == 'internal'):
            NSIDE = np.sqrt(N_pix/12).astype('int')            
            pixels = self.internal_ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])

        #bin data by pixel
        Emin, Emax = summary_properties['Emin'], summary_properties['Emax']
        N_energy_bins = summary_properties['N_energy_bins']
        if summary_properties['log_energy_bins'] is True:
            E_bins = np.logspace(np.log10(Emin), np.log10(Emax), num=N_energy_bins+1)
        else:
            E_bins = N_energy_bins
        
        #All photon energies
        photon_energies = photon_info['energies']

        # To do: implement batched photon info
        #print('constructing map')
        for batchi in range(N_batch):
            if len(photon_energies) == 0:
                continue
            #Histogram works inclusively on the lower edge, so this should work
            hist, pix_edges, E_edges = np.histogram2d(pixels, photon_energies, range = ((0,N_pix),(Emin, Emax)), bins = [N_pix,np.logspace(np.log10(Emin), np.log10(Emax),num=N_energy_bins+1)])

            if summary_properties['galactic_plane_latitude_cut'] is not None:
                gal_lat = summary_properties['galactic_plane_latitude_cut']
                colat, _ = hp.pix2ang(NSIDE, np.arange(0, N_pix))
                pixels_in_plane = (colat > np.pi / 2 - gal_lat) & (colat < np.pi / 2 + gal_lat)
                hist[pixels_in_plane] = hp.UNSEEN

            energy_dependent_map[batchi,:,:] = hist
        
        return energy_dependent_map 

    def get_energy_dependent_histogram_from_map(self, input_map, summary_properties):
        '''
        Takes in binned data (i.e a map with dimension N_pix x N_energy) and return a summary statistic
        '''
        #Properties of map
        #N_E should match what's in summary_properties
        N_batch, N_pix, N_E = input_map.shape

        #properties of histogram
        N_bins = summary_properties['histogram_properties']['Nbins']
        Cmin_hist, Cmax_hist = summary_properties['histogram_properties']['Cmin_hist'], summary_properties['histogram_properties']['Cmax_hist']
        #Cmax_hist can be array or scalar

        output_summary = np.zeros((N_batch, N_bins*N_E))
        for bi in range(0,N_batch):
            summary_bi = np.zeros((N_bins, N_E))
            for ei in range(0,N_E):

                if (np.isscalar(Cmax_hist)):
                    max_counts_value = Cmax_hist
                else:
                    max_counts_value = Cmax_hist[ei]

                #Compute histogram of batch bi and energy bin ei
                hist, bin_edges = np.histogram(input_map[bi,:,ei], bins = N_bins, range = (Cmin_hist, max_counts_value))

                #store the histogram
                summary_bi[:,ei] = hist

            #Output summary is flattened version of energy-dependent histogram
            output_summary[bi,:] = summary_bi.transpose().flatten()

        return output_summary

    #####################
    ##### UTILITIES #####
    #####################
    def _print_line(self, num=None):
        """Helper function for beautifying output.

        :param num: number of dashes to print in the output line
        :type num: int
        """
        if num is not None:
            print('-'*num)
        else:
            print('-'*self._terminal_columns)

    def print_output(self, message: str, *, kind: str, header: int = 0, footer: int = 0, prepend: bool = False):
        """Output and print a statement.
        
        This can be used to control messages 

        :param message: Message to be printed
        :type message: str
        :param kind: kind of message. Should be an attribute either 'verbose' or 'debug'
        :type kind: str, 'verbose' or 'debug'
        :param header: number of lines to print above message. 2 for prominent messages (different stages of analysis). 1 for updates or summaries. 0 for debug or minor output
        :type header: int
        :param prepend: 
        :type prepend: bool
        """
        # Print nothing if self.silent is True
        if getattr(self, "silent", 0):
            return None

        # Print and format according to kind
        if getattr(self, kind, 0):
            for _ in range(header):
                self._print_line()

            if prepend:
                message = '\t-->' + message

            print(message)

            for _ in range(footer):
                self._print_line()


