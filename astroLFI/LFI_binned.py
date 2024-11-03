import numpy as np
import pdb
import scipy
import scipy.interpolate
import scipy.integrate as integrate
import astropy.units as units
import healpy as hp
import torch

'''
The class is used to generate mock datasets, and to interface that generation with ELFI sampling methods.
'''

class LFI_binned:

    def __init__(self, parameter_range):
        #Number of parameters used to specify model
        self.N_parameters = len(parameter_range[0])

        #allowed ranges of parameters
        self.param_min = parameter_range[0]
        self.param_max = parameter_range[1]

        if (self.N_parameters != len(self.param_max)):
            raise NameError("Parameter range dimension mismatch detected")

        #Can contain an optional prior, beyond uniform prior
        #specified by param_min and param_max
        self.lnprior = None
                 
    def setup_binned(self, PDF_and_spectrum_list, is_isotropic_list, is_poisson_list, count_range, energy_range, num_energy_bins, solid_angle_pixel, verbose = False, args = None):
        '''
        A binned analysis is one in which we don't keep track of individual emiting objects.  Rather, for
        each source class we have a function that describes the probability of getting C photons in a spatial 
        bin (i.e. the PDF), and a function that describes the probability that that source class emits a photon of
        a particular energy (i.e. the spectrum).        
        '''
        self.analysis_type = 'binned'
        
        #PDF_model_list contains the PDF functions and spectrum functions
        #that represent our model
        self.PDF_spec = PDF_and_spectrum_list

        #is a source class istropic?
        self.is_istropic_list = is_isotropic_list

        #is a source class Poisson?
        self.is_poisson_list = is_poisson_list

        #Area of pixels on the sky
        self.solid_angle_pixel = solid_angle_pixel

        #Number of types of sources contributing photons
        self.N_source_classes = len(PDF_and_spectrum_list)

        #Extra arguments to pass to PDF and spectrum functions (these should be constant, i.e. independent of parameters)
        self.args = args
        
        #Energy binning
        self.Emin = energy_range[0]
        self.Emax = energy_range[1]
        self.nEbins = num_energy_bins
        self.Ebins = np.geomspace(self.Emin, self.Emax, self.nEbins+1)
                    
        #Count binning
        self.Cmin = count_range[0]
        self.Cmax = count_range[1]
        self.Cbins = np.arange(0, self.Cmax)

        self.verbose = verbose
        if (self.verbose):
            print("Analysis Type: " + self.analysis_type)
            print("N_parameters = ", self.N_parameters)
            print("Isotropic = ", self.isotropic)
            print("parameter min = ", self.param_min)
            print("parameter max = ", self.param_max)
            print("Emin = ", self.Emin)
            print("Emax = ", self.Emax)
            print("N_source_classes = ", self.N_source_classes)
            
    def add_lnprior(self, lnprior_func):
        # lnprior_func is prior on parameters
        self.lnprior = lnprior_func
        
        if (self.verbose):
            print("prior added")


    ##########################################################################
    '''
    Basic statistical functions

    '''
    ##########################################################################

    def sample_from_uniform(self, N_samples):
        # Generate random parameter samples drawn from uniform distribution given
        # by parameter ranges
        output_samples = np.zeros((N_samples, self.N_parameters))
        for ii in range(0,self.N_parameters):
            output_samples[:,ii] = np.random.uniform(low = self.param_min[ii],\
                                                     high = self.param_max[ii],\
                                                     size = N_samples)
        return output_samples
    
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
    
    ##########################################################################
    '''
    Mock data generation functions

    '''
    ##########################################################################
    
    def generate_mock_data_binned(self, input_params, exposure = 1.0, batch_size = None, N_pix = 1, verbose = False):
        #print("input params = ", input_params)
        '''
        Simulates data in map with N_pix pixels and N_energy energy bins
        '''
        
        # Array that will hold spatially and energy-binned photon counts
        if (batch_size is None):
            batch_size_to_use = 1
        else:
            batch_size_to_use = batch_size
        data_arr = np.zeros((batch_size_to_use, self.N_source_classes, N_pix, self.nEbins))
        
        #loop over source classes
        for si in range(0,self.N_source_classes):
            for bi in range(0,batch_size_to_use):
                if (batch_size is None):
                    params = input_params
                if (not batch_size is None):
                    params = np.array([input_params[ii][bi] for ii in range(0,self.N_parameters)])

                #save draws
                if (self.is_istropic_list[si]):
                    EPDF_draws = self.draw_from_isotropic_EPDF(params,si,\
                                                    exposure,self.solid_angle_pixel,N_pix )
                else:
                    print("NON ISOTROPIC CASE NOT IMPLEMENTED")
                data_arr[bi, si, :, :] = EPDF_draws
                

        #Now sum over sources to get final data
        data_arr_out = np.sum(data_arr, axis = 1)

        if (verbose):
            print(data_arr_out)
        
        return data_arr_out
                
    ##########################################################################
    '''
    Summary statistic functions

    '''
    ##########################################################################

    def get_summary_from_binned(self, input_data, summary_properties = {'type': 'None' ,'Nbins':10, 'Cmax_hist': 10, 'Cmin_hist': 0, 'energy_bins_to_use':'all'}):
        '''
        Takes in binned data (i.e a map with dimension N_pix x N_energy) and return a summary statistic
        '''
        if (summary_properties['type'] == 'None'):
            output_summary = np.copy(input_data)
        if (summary_properties['type'] == 'histogram'):
            N_batch, N_pix, N_E = input_data.shape[0], input_data.shape[1], input_data.shape[2]
            if (np.isscalar(summary_properties['energy_bins_to_use'])):
                N_E_summary = N_E
            else:
                N_E_summary = len(summary_properties['energy_bins_to_use'])
            output_summary = np.zeros((N_batch, summary_properties['Nbins']*N_E_summary))
            for bi in range(0,N_batch):
                temp = np.zeros((summary_properties['Nbins'], N_E_summary))
                for ei in range(0,N_E_summary):
                    if (np.isscalar(summary_properties['energy_bins_to_use'])):
                        energy_index = ei
                    else:
                        energy_index = summary_properties['energy_bins_to_use'][ei]

                    if (np.isscalar(summary_properties['Cmax_hist'])):
                        max_counts_value = summary_properties['Cmax_hist']
                    else:
                        max_counts_value = summary_properties['Cmax_hist'][ei]

                    #Should throw error if requesting log bins and minumum
                    #counts is zero
                    if (summary_properties['logbins']):
                        if (summary_properties['Cmin_hist'] == 0):
                            bins_to_use = np.append([-0.00001],np.exp(np.linspace(0.,np.log(max_counts_value), num = summary_properties['Nbins'])))
                        else:
                            bins_to_use = np.exp(np.linspace(np.log(summary_properties['Cmin_hist']),np.log(max_counts_value), num = summary_properties['Nbins']+1))
                        hist, bin_edges = np.histogram(input_data[bi,:,energy_index], bins = bins_to_use)
                    else:
                        hist, bin_edges = np.histogram(input_data[bi,:,energy_index], bins = summary_properties['Nbins'], range = (summary_properties['Cmin_hist'], max_counts_value))

                    temp[:,energy_index] = hist
                output_summary[bi,:] = temp.transpose().flatten()
        return output_summary
    
    ##########################################################################
    '''
    Distance functions

    '''
    ##########################################################################    
    
    def compute_binned_distance(self, vec1, vec2):
        '''
        ABC needs a distance metric to compare real data and simulated data.
        '''
        distance = np.sum((vec1 - vec2)**2.)
        return distance  

    ##########################################################################
    '''
    Functions that run ABC

    '''
    ##########################################################################    
            
    def run_abc_binned(self, input_data, N_samples, summary_properties, batch_size = None, epsilon = 100., verbose = False):
        '''
        Start the ABC sampling process
        '''
        N_pix, N_E_bins = input_data.shape[0], input_data.shape[1]

        # Add in batch dimension so that get summary works
        input_data_dim = np.expand_dims(input_data, axis = 0)
        
        # Convert input data to summary statistic
        input_summary = self.get_summary_from_binned(input_data_dim, summary_properties)

        #Draw a bunch of random parameter points from priors
        uniform_samples = self.sample_from_uniform(N_samples)

        #Will keep track of distances for all samples
        distances = np.zeros(N_samples)

        print_num = int(np.floor(N_samples/10.))
        for ii in range(0,N_samples):
            if (ii % print_num == 0 and verbose):
                print("sample = ", ii, " out of ", N_samples)
                
            #For each point in parameter space generate a mock data set
            mock_data = self.generate_mock_data_binned(uniform_samples[ii,:], batch_size = batch_size, N_pix = N_pix)

            #Convert mock data to summary
            mock_summary = self.get_summary_from_binned(mock_data, summary_properties)

            #Compute the distance between each mock data set and the input data
            distances[ii] = self.compute_binned_distance(mock_summary, input_summary)
        if verbose:
            print(distances)

        #Determine which mock data set realizations are within some minimum distance of the input data
        good_distance = np.where(distances < epsilon)[0]
        
        #return remaining parameter samples
        posterior_samples = uniform_samples[good_distance,:]

        return posterior_samples

    ##########################################################################
    '''
    ELFI functions

    '''
    ##########################################################################
    
    #Wrapper for using ELFI
    def generate_mock_data_binned_ELFI(*input, N_pix = 1, epdf_object = None, batch_size = 1, random_state=None):
        output = np.zeros((batch_size, N_pix, epdf_object.nEbins))
        #This should be list of length batch_size
        param_list = input[1:] #gets rid of self
        output = epdf_object.generate_mock_data_binned(param_list, batch_size = batch_size, N_pix = N_pix)
        return output

    def get_ELFI_model():
        m = elfi.new_model()
        priors = []
        for pi in range(0,len(param_min)):
            p1 = elfi.Prior('uniform', param_min[0], param_max[0])
        p2 = elfi.Prior('uniform', param_min[1], param_max[1])
        p3 = elfi.Prior('uniform', param_min[2], param_max[2])
        p4 = elfi.Prior('uniform', param_min[3], param_max[3])
        y0 = my_abc.generate_mock_data_binned_ELFI(param_true[0], param_true[1], param_true[2], param_true[3], N_pix = N_pix, epdf_object = my_abc)
        print("y0 generated")
        
        from functools import partial
    
        fn_simulator = partial(my_abc.generate_mock_data_binned_ELFI, N_pix = N_pix, epdf_object=my_abc)
        sim = elfi.Simulator(fn_simulator, p1, p2, p3, p4, observed=y0)

        fn_summary = partial(my_abc.get_summary_from_binned)
        #fn_summary = partial(my_abc.get_summary_from_binned_ELFI, epdf_object=my_abc)
        #testa = my_abc.get_summary_from_binned(y0)
        #test = fn_summary(y0)
        #pdb.set_trace()
        S1 = elfi.Summary(fn_summary, sim)
        
        d = elfi.Distance('euclidean', S1)

        rej = elfi.Rejection(d, batch_size=1)
        res = rej.sample(1000, threshold=50)

    ##########################################################################
    '''
    Unbinned analysis functions

    '''
    ##########################################################################
    
    def setup_unbinned(self, abundance_luminosity_and_spectrum_list, is_isotropic_list, is_background_list, energy_range, luminosity_range, radius_range, exposure, lat_cut = 0, verbose = False):
        self.analysis_type = 'unbinned'
        
        self.GC_to_earth = 8.5 #kpc
        
        #PDF_model_list contains the PDF functions and spectrum functions
        #that represent our model
        self.abun_lum_spec = abundance_luminosity_and_spectrum_list
        
        #is a source class istropic?
        self.is_isotropic_list = is_isotropic_list
        
        #is a source class background?
        self.is_background_list = is_background_list

        #allowed mass range
        self.Lmin = luminosity_range[0]
        self.Lmax = luminosity_range[1]

        #galactic latitude outside of which to generate photons
        self.lat_cut = lat_cut
        
        #allowed minimum/maximum distance from galactic center
        self.Rmin = radius_range[0]
        self.Rmax = radius_range[1]
        if self.lat_cut != 0:
            self.Rmin = self.GC_to_earth*np.sin(np.radians(lat_cut))
            if verbose:
                print('Rmin changed to match latitude cut')
        
        #allowed energy range
        self.Emin = energy_range[0]
        self.Emax = energy_range[1]

        #is a source class Poisson?
        #self.is_poisson_list = is_poisson_list
        
        #exposure of detector (converted from cm^2yr to kpc^2s)
        self.exposure = exposure*(units.cm.to('kpc')**2)*units.yr.to('s')

        #Number of types of sources contributing photons
        self.N_source_classes = len(abundance_luminosity_and_spectrum_list)

        self.verbose = verbose
        if (self.verbose):
            print("Analysis Type: " + self.analysis_type)
            print("N_parameters = ", self.N_parameters)
            print("Isotropic = ", self.is_isotropic_list)
            print("parameter min = ", self.param_min)
            print("parameter max = ", self.param_max)
            print("Emin = ", self.Emin)
            print("Emax = ", self.Emax)
            print("Rmin = ", self.Rmin)
            print("Rmax = ", self.Rmax)
            print("Lmin = ", self.Lmin)
            print("Lmax = ", self.Lmax)
            print("exposure = ", self.exposure)
            print("lat_cut = ", self.lat_cut)
            print("N_source_classes = ", self.N_source_classes)

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
        
        return photon_info

    def create_sources(self, input_params, grains = 1000):
        '''
        This function creates a list of sources, where each source has a radial distance, mass, and luminosity

        dndM = number density of sources in infinitessimal mass bin
        input and output masses are in units of the solar mass
        maxr = max radial distance from galactic center in kpc
        '''
        source_info = {}

        #Loop over all source types
        for si in range(self.N_source_classes):
            if self.is_background_list[si]:
                continue
            if self.is_isotropic_list[si]:
                # Draw masses and radii for each source from abundance
                luminosities, radii, single_p_radii  = self.draw_luminosities_and_radii(input_params, self.abun_lum_spec[si][0], grains=grains)
                num_sources = np.size(luminosities)
                num_single_p_sources = np.size(single_p_radii)

                # Draw angles for every source [theta, phi]
                angles = np.ones([num_sources, 2])
                angles[:,0] = np.arccos(1 - 2*np.random.rand(num_sources))
                angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num_sources)
                single_p_angles = np.ones([num_single_p_sources, 2])
                single_p_angles[:,0] = np.arccos(1 - 2*np.random.rand(num_single_p_sources))
                single_p_angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num_single_p_sources)
            else:
                masses, radii, angles = self.draw_masses_radii_angles(input_params, self.abun_lum_spec[si][0], self.abun_lum_spec[si][1])
                num_sources = np.size(masses)
            
            # Get the distance from earth to each source
            distances = np.sqrt((self.GC_to_earth + radii*np.cos(angles[:,0]))**2 + (radii*np.sin(angles[:,0]))**2)
            single_p_distances = np.sqrt((self.GC_to_earth + single_p_radii*np.cos(single_p_angles[:,0]))**2 + (single_p_radii*np.sin(single_p_angles[:,0]))**2)
            if self.is_isotropic_list[si]:
                prob_factors = ((self.GC_to_earth - single_p_radii)/single_p_distances)**2
                bad_source_indices = np.where(np.random.rand(num_single_p_sources) > prob_factors)
                single_p_radii = np.delete(single_p_radii, bad_source_indices)
                single_p_angles = np.delete(single_p_angles, bad_source_indices, axis = 0)
                single_p_distances = np.delete(single_p_distances, bad_source_indices)
                num_single_p_sources = np.size(single_p_radii)
    
            # Get the angles from earth to each source
            earth_angles = np.ones([num_sources, 2])
            earth_angles[:,0] = np.arccos((self.GC_to_earth + radii*np.cos(angles[:,0]))/distances)
            earth_angles[:,1] = angles[:,1]
            single_p_earth_angles = np.ones([num_single_p_sources, 2])
            single_p_earth_angles[:,0] = np.arccos((self.GC_to_earth + single_p_radii*np.cos(single_p_angles[:,0]))/single_p_distances)
            single_p_earth_angles[:,1] = single_p_angles[:,1]
    
            # Convert earth angles to angles such that the GC is at theta=pi/2
            new_earth_angles = np.ones([num_sources, 2])
            new_earth_angles[:,0] = np.arccos(-np.sin(earth_angles[:,0])*np.cos(earth_angles[:,1]))
            new_earth_angles[:,1] = (np.arctan(np.tan(earth_angles[:,0])*np.sin(earth_angles[:,1])) + 
                                     np.pi*((1-np.sign(np.cos(earth_angles[:,0])))/2))%(2*np.pi)
            single_p_new_earth_angles = np.ones([num_single_p_sources, 2])
            single_p_new_earth_angles[:,0] = np.arccos(-np.sin(single_p_earth_angles[:,0])*np.cos(single_p_earth_angles[:,1]))
            single_p_new_earth_angles[:,1] = (np.arctan(np.tan(single_p_earth_angles[:,0])*np.sin(single_p_earth_angles[:,1])) + 
                                     np.pi*((1-np.sign(np.cos(single_p_earth_angles[:,0])))/2))%(2*np.pi)
            
            # Remove sources inside latitude cut
            keep_i = np.where(np.abs(new_earth_angles[:,0] - np.pi/2) > np.radians(self.lat_cut))[0]
            single_p_keep_i = np.where(np.abs(single_p_new_earth_angles[:,0] - np.pi/2) > np.radians(self.lat_cut))[0]
            num_sources = np.size(keep_i)
            single_p_num_sources = np.size(single_p_keep_i)
            luminosities = luminosities[keep_i]
            distances = distances[keep_i]
            single_p_distances = single_p_distances[single_p_keep_i]
            new_earth_angles = new_earth_angles[keep_i,:]
            single_p_new_earth_angles = single_p_new_earth_angles[single_p_keep_i,:]
            
            # Catalog the type of source
            types = si*np.ones(num_sources)
            single_p_types = si*np.ones(num_single_p_sources)
            
            if si == 0:
                source_info = {'luminosities': luminosities,
                               'distances': distances,
                               'single_p_distances': single_p_distances,
                               'angles': new_earth_angles,
                               'single_p_angles': single_p_new_earth_angles,
                               'types': types,
                               'single_p_types' : single_p_types}
            else:
                source_info = {'luminosities':np.concatenate((source_info['luminosities'], luminosities)),
                               'distances':np.concatenate((source_info['distances'], distances)),
                               'single_p_distances':np.concatenate((source_info['single_p_distances'], single_p_distances)),
                               'angles':np.concatenate((source_info['angles'], new_earth_angles)),
                               'single_p_angles':np.concatenate((source_info['single_p_angles'], single_p_new_earth_angles)),
                               'types': np.concatenate((source_info['types'], types)),
                               'single_p_types': np.concatenate((source_info['single_p_types'], single_p_types))}

        if (self.verbose):
            print(source_info)
        
        return source_info
    

    def generate_photons_from_sources(self, input_params, source_info):
        '''
        Function returns list of photon energies and sky positions
        '''
        # Calculate mean expected flux from each source
        mean_photon_counts = self.exposure*source_info['luminosities']/(4.*np.pi*source_info['distances']**2.)

        # Poisson draw from mean photon counts to get realization of photon counts
        photon_counts = np.random.poisson(mean_photon_counts)

        # List of angles of photons
        photon_angles = np.ones([np.sum(photon_counts), 2])
        photon_angles[:,0] = np.repeat(source_info['angles'][:,0], photon_counts)
        photon_angles[:,1] = np.repeat(source_info['angles'][:,1], photon_counts)
        photon_angles = np.concatenate((photon_angles, source_info['single_p_angles']))

        # Assign energies to all of those photons
        energies = np.array([])
        single_p_energies = np.array([])
        for si in range(self.N_source_classes):
            if self.is_background_list[si]:
                continue
            type_indices_bool = np.where(source_info['types'] == si, True, False)
            num_single_p_sources = np.size(np.where(source_info['single_p_types'] == si, True, False))
            num_E = int(round(self.Emax - self.Emin))
            energy_array = np.linspace(self.Emin, self.Emax, num_E + 1)
            dndE = self.abun_lum_spec[si][1](input_params, energy_array)
            if type(dndE) == type(torch.tensor([])):
                dndE = dndE.numpy()
            energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                dndE/np.sum(dndE),
                                                np.sum(photon_counts, where = type_indices_bool))
            single_p_energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                dndE/np.sum(dndE),
                                                num_single_p_sources)
            energies = np.concatenate((energies, energy_array[energy_indices]))
            single_p_energies = np.concatenate((single_p_energies, energy_array[single_p_energy_indices]))
        energies = np.concatenate((energies, single_p_energies))
        for si in range(self.N_source_classes):
            if not self.is_background_list[si]:
                continue
            if self.is_isotropic_list[si]:
                num_E = int(round(self.Emax - self.Emin))
                energy_array = np.linspace(self.Emin, self.Emax, num_E + 1)
                dndE = self.abun_lum_spec[si][0](input_params, energy_array)
                if type(dndE) == type(torch.tensor([])):
                    dndE = dndE.numpy()
                solid_angle = 2*np.pi*2*(1-np.cos(np.radians(90. - self.lat_cut)))
                N_iso_photons = np.random.poisson(np.round(np.sum(dndE*self.exposure*(units.kpc.to('cm')**2)*solid_angle)).astype('int'))
                As = np.ones([N_iso_photons, 2])
                As[:,0] = np.arccos(1 - np.random.rand(N_iso_photons))
                As[:,0] += (np.pi - 2*As[:,0])*np.random.randint(2, size = N_iso_photons)
                As[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = N_iso_photons)
                if type(dndE) == type(torch.tensor([])):
                    dndE = dndE.numpy()
                energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                    dndE/np.sum(dndE), N_iso_photons)
                photon_angles = np.concatenate((photon_angles, As))
                energies = np.concatenate((energies, energy_array[energy_indices]))
            else:
                As, Es = self.draw_angles_and_energies(self.abun_lum_spec[si][0], self.abun_lum_spec[si][1])
                photon_angles = np.concatenate((photon_angles, As))
                energies = np.concatenate((energies, Es))
            
        photon_info = {'angles':photon_angles, 'energies':energies}

        if (self.verbose):
            print(photon_info)
        
        return photon_info        
    
    def get_map_from_unbinned(self, photon_info, N_pix, N_energy, map_type = 'healpix'):
        '''
        Given unbinned photon data, return maps with dimension npix x N_energy

        map_type can be healpix or internal
        '''
        
        #The output map
        output_summary = np.zeros((N_pix, N_energy))

        if (map_type == 'healpix'):
            NSIDE = np.sqrt(N_pix/12).astype('int')
            pixels = hp.ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])
        elif (map_type == 'internal'):
            NSIDE = np.sqrt(N_pix/12).astype('int')            
            pixels = self.internal_ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])

        #bin data by pixel
        for pix_index in range(N_pix):
            pix_energies = photon_info['energies'][np.where(pixels == pix_index)]
            hist, hist_edges = np.histogram(pix_energies, bins = N_energy, range = (self.Emin, self.Emax))
            output_summary[pix_index,:] = hist
        
        return  output_summary

    '''
    #used in following get_map_from_unbinned function, works identically to healpix version but with different
    #spherical partition
    def internal_ang2pix(self, NSIDE, data_thetas, data_phis):
        thetas = np.arccos(1 - np.linspace(0, 2*NSIDE, 2*NSIDE + 1)/NSIDE)
        phis = np.linspace(0, 2*np.pi, NSIDE + 1)
        data_theta_index = np.searchsorted(thetas, data_thetas)-1
        data_phi_index = np.searchsorted(phis, data_phis)-1
        data_index = NSIDE*data_theta_index + data_phi_index

        return data_index
    '''
    
    def draw_from_isotropic_background_unbinned(self, Ebins, exposure, Sangle):
        e, dnde = self.e_isotropic, self.dnde_isotropic
        f = scipy.interpolate.interp1d(e, dnde, kind='linear', fill_value=0.)
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

    ##########################################################################
    '''s
    New code for Fermi analysis
    '''
    ##########################################################################
    
    #for isotropic adundances
    def draw_luminosities_and_radii(self, input_params, abundance, N_draws = 0, grains = 1000, epsilon = 0.1):
        lumVals = np.geomspace(self.Lmin, self.Lmax, grains)
        #radiusVals = np.geomspace(self.Rmin, self.Rmax, grains)
        #lumVals = np.linspace(self.Lmin, self.Lmax, grains)
        radiusVals = np.linspace(self.Rmin, self.Rmax, grains)
        PDF = abundance(input_params, np.tile(lumVals[:-1],(grains-1,1)).T, np.tile(radiusVals[:-1],(grains-1,1)))
        dL = np.tile(lumVals[1:]-lumVals[:-1],(grains-1,1)).T
        dR = np.tile(radiusVals[1:]-radiusVals[:-1],(grains-1,1))
        integrand = PDF*4*np.pi*(np.tile(radiusVals[:-1],(grains-1,1)))**2*dL*dR
        #binomially draw low luminosity sources to save computation time
        Dconserv = np.abs(self.GC_to_earth - np.tile(radiusVals[:-1],(grains-1,1)))
        Dconserv = np.where(Dconserv == 0, 0.00000000001, Dconserv)
        C = (np.tile(lumVals[:-1],(grains-1,1)).T)*self.exposure/(4*np.pi*Dconserv**2)
        Ci = np.where(C < epsilon)
        C = np.where(C < epsilon, C, 0)
        p = C*np.exp(-C)
        num_single_p_sources_at_radii = np.round(np.sum(integrand*p, axis = 0)).astype(int)
        single_p_radii = np.repeat(radiusVals[:-1], num_single_p_sources_at_radii)
        integrand[Ci] = 0
        lum_indices, radus_indices = self.draw_from_2D_pdf(integrand, N_draws)
        return lumVals[lum_indices], radiusVals[radus_indices], single_p_radii
    
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
    
    #Moffat function used in PSF
    def King(x, sigma, gamma):
        return(1/(2*np.pi*sigma**2))*(1-(1/gamma))*(1+(1/(2*gamma))*(x**2/sigma**2))**(-gamma)
    
    def PSF_energy_dispersion(self, photon_info, angle_res, energy_res):
        num_photons = np.size(photon_info['energies'])
        '''
        #old method of approximating surface of sphere as flat and dropping a 2d-gaussian on it, then smear angles
        distances = np.sqrt(-2*angle_res**2*np.log(1-np.random.random(num_photons)))
        '''
        C0 = 6.38e-2
        C1 = 1.26e-3
        beta = 0.8
        #parameters below are not correct, need PSF FITS file
        Ntail = 1
        Score = 1
        Gcore = 1
        Stail = 1
        Gtail = 1
        Fcore = 1/(1 + Ntail*Stail**2/Score**2)
        x_vals = np.linspace(0, 10, 1000)
        PSF = Fcore*self.King(x_vals, Score, Gcore) + (1-Fcore)*self.King(x_vals, Stail, Gtail)
        x = self.draw_from_pdf(x_vals, PSF*2*np.pi*x_vals/np.sum(PSF*2*np.pi*x_vals), num_photons)
        S_p = np.sqrt((C0*(photon_info['energies']/100)**(-beta))**2 + C1**2)
        distances = 2*np.sin(x*S_p/2)
        rotations = 2*np.pi*np.random.random(num_photons)
        delta_thetas = distances*np.cos(rotations)
        delta_phis = distances*np.sin(rotations)
        photon_info['angles'][:,0] += delta_thetas
        over_indices = np.where(np.logical_or(photon_info['angles'][:,0] > np.pi, photon_info['angles'][:,0] < 0))
        photon_info['angles'][over_indices,0] = np.pi - photon_info['angles'][over_indices,0]%np.pi
        photon_info['angles'][over_indices,1] += np.pi
        photon_info['angles'][:,1] += delta_phis
        photon_info['angles'][:,1] %= 2*np.pi
        #smear energies
        sig = energy_res/(2*np.sqrt(2*np.log(2))) #assumes energy_res is FWHM of gaussian. sig*energy is then the standard deviation
        photon_info['energies'] = np.random.normal(photon_info['energies'], sig*photon_info['energies'])
        return photon_info
    
    '''
    #for non-isotropic abundances (requires large RAM)
    def draw_masses_radii_angles(self, input_params, abundance, luminosity, N_draws = 0, grains = 100):
        #intensityLim = 0.1#expected photons/bin <--not sure if this is justified, so turned off
        block = int(grains/10)
        massVals = np.geomspace(self.Mmin, self.Mmax, grains+1)
        radiusVals = np.geomspace(self.Rmin, self.Rmax, grains+1)
        thetaVals = np.linspace(0,np.pi,grains+1)
        phiVals = np.linspace(0,2*np.pi,grains+1)
        massPDF, radiusPDF, thetaPDF, phiPDF = np.zeros(grains), np.zeros(grains), np.zeros(grains), np.zeros(grains)
        for i in np.linspace(0, grains-block, int(grains/block)).astype(int):
            for j in np.linspace(0, grains-block, int(grains/block)).astype(int):
                for k in np.linspace(0, grains-block, int(grains/block)).astype(int):
                    massArray = np.tile(massVals[1:],(block,block,block,1)).T
                    radiusArray = np.tile(np.tile(radiusVals[1+i:1+i+block],(grains,1)).T,(block,block,1,1)).T
                    thetaArray = np.tile(np.tile(thetaVals[1+j:1+j+block],(block,1)).T,(grains,block,1,1))
                    phiArray = np.tile(phiVals[1+k:1+k+block],(grains,block,block,1))
                    PDF = abundance(input_params, massArray, radiusArray, thetaArray, phiArray)
                    del phiArray
                    luminosityArray, sigArray = luminosity(input_params, massArray, radiusArray)
                    del massArray, sigArray
                    dM = np.tile(massVals[1:]-massVals[:-1],(block,block,block,1)).T
                    dR = np.tile(np.tile(radiusVals[1+i:1+i+block]-radiusVals[i:i+block],(grains,1)).T,(block,block,1,1)).T
                    dT = thetaVals[1]-thetaVals[0]
                    dP = phiVals[1]-phiVals[0]
                    integrand = PDF*radiusArray**2*np.sin(thetaArray)*dM*dR*dT*dP
                    del PDF, dM, dR
                    distanceArray = np.sqrt((self.GC_to_earth + radiusArray*np.cos(thetaArray))**2 + (radiusArray*np.sin(thetaArray))**2)
                    del radiusArray, thetaArray
                    #intensityArray = integrand*luminosityArray*self.exposure/(4*np.pi*distanceArray**2) <--not sure if this is justified, so turned off
                    integrand = np.where(intensityArray < intensityLim, 0, integrand)
                    massPDF += np.sum(integrand, axis = (1,2,3))
                    radiusPDF[i:i+block] += np.sum(integrand, axis = (0,2,3))
                    thetaPDF[j:j+block] += np.sum(integrand, axis = (0,1,3))
                    phiPDF[k:k+block] += np.sum(integrand, axis = (0,1,2))
        print('pdfs')
        if N_draws == 0:
            N_draws = int(round(np.sum(massPDF)))
        print(N_draws)
        masses = massVals[self.draw_from_pdf(massVals[1:], massPDF/np.sum(massPDF), N_draws)]
        radii = massVals[self.draw_from_pdf(massVals[1:], massPDF/np.sum(massPDF), N_draws)]
        angles = np.ones((N_draws, 2))
        angles[:,0] = thetaVals[self.draw_from_pdf(thetaVals[1:], thetaPDF/np.sum(thetaPDF), N_draws)]
        angles[:,1] = phiVals[self.draw_from_pdf(phiVals[1:], phiPDF/np.sum(phiPDF), N_draws)]
        return masses, radii, angles
    
    #returns N_draws angles with granularity set by grains. Takes angular density(theta, phi)
    def draw_angles_from_density(self, density, N_draws, grains = 10000):
        thetaVals = np.linspace(0,np.pi,grains)
        phiVals = np.linspace(0,2*np.pi,grains)
        PDF = density(np.tile(thetaVals[1:],(grains-1,1)).T, np.tile(phiVals[1:],(grains-1,1)))
        integrand = PDF*(np.tile(np.sin(thetaVals[1:]),(grains-1,1)).T)
        thetaPDF = np.sum(integrand, axis = 1)
        phiPDF = np.sum(integrand, axis = 0)
        angles = np.ones((N_draws, 2))
        angles[:,0] = thetaVals[self.draw_from_pdf(thetaVals[1:], thetaPDF/np.sum(thetaPDF), N_draws)]
        angles[:,1] = phiVals[self.draw_from_pdf(phiVals[1:], phiPDF/np.sum(phiPDF), N_draws)]

        return angles
    '''
    
    ##########################################################################
    '''
    Old sampling stuff.  Possibly not needed anymore if we're using ELFI.
    '''
    ##########################################################################
    
    def run_abc_unbinned(self, input_data, N_samples, N_energy, epsilon = 100.):

        #Draw a bunch of random parameter points from priors
        uniform_samples = self.sample_from_uniform(N_samples)

        # Convert input data to summary statistic
        input_summary = self.unbinned_summary_statistic(input_data, N_energy)

        #Will keep track of distances for all samples
        distances = np.zeros(N_samples)

        for si in range(N_samples):
            mock_data = self.generate_mock_data_unbinned(uniform_samples[si])
            mock_summary = self.unbinned_summary_statistic(mock_data, N_energy)
            distances[si] = self.compute_binned_distance(mock_summary, input_summary)

        #Determine which mock data set realizations are within some minimum distance of the input data
        good_distance = np.where(distances < epsilon)
        
        #return remaining parameter samples
        posterior_samples = uniform_samples[good_distance,:]
        
        return posterior_samples
