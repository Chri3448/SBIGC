import numpy as np
import pdb
import scipy
import scipy.interpolate
import scipy.integrate as integrate
import astropy.units as units
import healpy as hp

'''
The class is used to generate mock datasets, and to interface that generation with ELFI sampling methods.
'''

class EPDFABC:

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
        x_pdf = np.sum(pdf, axis = 1)/np.sum(pdf)
        x_cdf = np.cumsum(x_pdf)
        x_rands = np.random.rand(Ndraws)
        x_indices = np.searchsorted(x_cdf, x_rands)
        y_cdfs = np.cumsum(pdf, axis = 1)/np.tile(np.sum(pdf, axis = 1), (np.size(pdf[:,0]),1)).T
        y_rands = np.random.rand(Ndraws)
        y_indices = np.zeros(np.size(x_indices), dtype = 'int')
        for i in range(np.size(pdf[:,0])):
            source_positions = np.where(x_indices == i)
            y_indices[source_positions] = np.searchsorted(y_cdfs[i,:], y_rands[source_positions])
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
    
    def setup_unbinned(self, abundance_luminosity_and_spectrum_list, is_isotropic_list, energy_range, mass_range, radius_range, exposure, verbose = False):
        self.analysis_type = 'unbinned'
        
        #PDF_model_list contains the PDF functions and spectrum functions
        #that represent our model
        self.abun_lum_spec = abundance_luminosity_and_spectrum_list
        
        #is a source class istropic?
        self.is_isotropic_list = is_isotropic_list

        #allowed energy range
        self.Emin = energy_range[0]
        self.Emax = energy_range[1]

        #allowed mass range
        self.Mmin = mass_range[0]
        self.Mmax = mass_range[1]

        #allowed maximum distance from galactic center
        self.Rmin = radius_range[0]
        self.Rmax = radius_range[1]

        #is a source class Poisson?
        #self.is_poisson_list = is_poisson_list
        
        #exposure of detector (converted from cm^2yr to kpc^2s)
        self.exposure = exposure*(units.cm.to('kpc')**2)*units.yr.to('s')
        
        self.GC_to_earth = 8.5 #kpc

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
            print("Mmin = ", self.Mmin)
            print("Mmax = ", self.Mmax)
            print("Rmin = ", self.Rmin)
            print("Rmax = ", self.Rmax)
            print("exposure = ", self.exposure)
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

    def create_sources(self, input_params):
        '''
        This function creates a list of sources, where each source has a radial distance, mass, and luminosity

        dndM = number density of sources in infinitessimal mass bin
        input and output masses are in units of the solar mass
        maxr = max radial distance from galactic center in kpc
        '''
        source_info = {}

        #Loop over all source types
        for si in range(self.N_source_classes):
            '''
            # Draw masses for every source
            num_mass = 1000            
            mass_array = np.geomspace(self.Mmin, self.Mmax, num_mass)
            dm = mass_array[1:] - mass_array[:-1]
            dndm = self.abun_lum_spec[si][0](input_params, mass_array)
            source_number_density = np.sum(dndm[1:]*dm) # the number of sources per volume
            num_sources = int(round((4./3.)*np.pi*(self.Rmax**3.)*source_number_density)) # the total number of sources
            print("num sources = ", num_sources)
            mass_indices = self.draw_from_pdf(np.arange(0,len(mass_array)), dndm/np.sum(dndm), num_sources)
            masses = mass_array[mass_indices]

            # Draw radii for every source (radii here is to galactic center)
            radii = np.random.uniform(low = 0., high = self.Rmax, size = num_sources)
            '''
            if self.is_isotropic_list[si]:
                # Draw masses and radii for each source from abundance
                masses, radii = self.draw_masses_and_radii(input_params, self.abun_lum_spec[si][0])
                num_sources = np.size(masses)

                # Draw angles for every source [theta, phi]
                angles = np.ones([num_sources, 2])
                angles[:,0] = np.arccos(1 - 2*np.random.rand(num_sources))
                angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num_sources)
            else:
                masses, radii, angles = self.draw_masses_radii_angles(input_params, self.abun_lum_spec[si][0], self.abun_lum_spec[si][1])
                num_sources = np.size(masses)
                #print('gen')
            
            # Get the distance from earth to each source
            distances = np.sqrt((self.GC_to_earth + radii*np.cos(angles[:,0]))**2 + (radii*np.sin(angles[:,0]))**2)
            #print('dte')
    
            # Get the angles from earth to each source
            earth_angles = np.ones([num_sources, 2])
            earth_angles[:,0] = np.arccos((self.GC_to_earth + radii*np.cos(angles[:,0]))/distances)
            earth_angles[:,1] = angles[:,1]
            #print('ea')
    
            # Convert earth angles to angles such that the GC is at theta=pi/2
            new_earth_angles = np.ones([num_sources, 2])
            new_earth_angles[:,0] = np.arccos(-np.sin(earth_angles[:,0])*np.cos(earth_angles[:,1]))
            new_earth_angles[:,1] = (np.arctan(np.tan(earth_angles[:,0])*np.sin(earth_angles[:,1])) + 
                                     np.pi*((1-np.sign(np.cos(earth_angles[:,0])))/2))%(2*np.pi)
            #print('nea')

            # Draw luminosities for every source
            meanL, sig = self.abun_lum_spec[si][1](input_params, masses, radii)
            luminosities = np.exp(np.random.normal(meanL, sig))
            #print('lum')

            # Catalog the type of source
            types = si*np.ones(num_sources)
            #print('type')
            
            if si == 0:
                source_info = {'masses': masses,
                               'radii': radii,
                               'distances': distances,
                               'angles': new_earth_angles,
                               'luminosities': luminosities,
                               'types': types}
            else:
                source_info = {'masses':np.concatenate((source_info['masses'], masses)),
                               'radii': np.concatenate((source_info['radii'], radii)),
                               'distances':np.concatenate((source_info['distances'], distances)),
                               'angles':np.concatenate((source_info['angles'], new_earth_angles), axis = 0),
                               'luminosities':np.concatenate((source_info['luminosities'], luminosities)),
                               'types': np.concatenate((source_info['types'], types))}

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

        # Assign energies to all of those photons
        energies = np.array([])
        for si in range(self.N_source_classes):
            type_indices_bool = np.where(source_info['types'] == si, True, False)
            num_E = int(round(self.Emax - self.Emin))
            energy_array = np.linspace(self.Emin, self.Emax, num_E)
            dE = energy_array[1:] - energy_array[:-1]
            dndE = self.abun_lum_spec[si][2](input_params, energy_array)
            energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                dndE/np.sum(dndE),
                                                np.sum(photon_counts, where = type_indices_bool))
            energies = np.concatenate((energies, energy_array[energy_indices]))
        
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


    #used in following get_map_from_unbinned function, works identically to healpix version but with different
    #spherical partition
    def internal_ang2pix(self, NSIDE, data_thetas, data_phis):
        thetas = np.arccos(1 - np.linspace(0, 2*NSIDE, 2*NSIDE + 1)/NSIDE)
        phis = np.linspace(0, 2*np.pi, NSIDE + 1)
        data_theta_index = np.searchsorted(thetas, data_thetas)-1
        data_phi_index = np.searchsorted(phis, data_phis)-1
        data_index = NSIDE*data_theta_index + data_phi_index

        return data_index

    
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
    '''
    New code for Fermi analysis
    '''
    ##########################################################################
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

    #for isotropic adundances
    def draw_masses_and_radii(self, input_params, abundance, N_draws = 0, grains = 1000):
        massVals = np.geomspace(self.Mmin, self.Mmax, grains)
        radiusVals = np.geomspace(self.Rmin, self.Rmax, grains)
        PDF = abundance(input_params, np.tile(massVals[:-1],(grains-1,1)).T, np.tile(radiusVals[:-1],(grains-1,1)))
        dM = np.tile(massVals[1:]-massVals[:-1],(grains-1,1)).T
        dR = np.tile(radiusVals[1:]-radiusVals[:-1],(grains-1,1))
        integrand = PDF*4*np.pi*(np.tile(radiusVals[:-1],(grains-1,1)))**2*dM*dR
        '''
        massPDF = np.sum(integrand, axis = 1)
        radiusPDF = np.sum(integrand, axis = 0)
        if N_draws == 0:
            N_draws = int(round(np.sum(massPDF)))
        masses = massVals[self.draw_from_pdf(massVals[:-1], massPDF/np.sum(massPDF), N_draws)]
        radii = radiusVals[self.draw_from_pdf(radiusVals[:-1], radiusPDF/np.sum(radiusPDF), N_draws)]
        '''
        mass_indices, radus_indices = self.draw_from_2D_pdf(integrand)
        #return masses, radii
        return massVals[mass_indices], radiusVals[radus_indices]
    
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

    def PSF_energy_dispersion(self, photon_info, angle_res, energy_res):
        #approximating surface of sphere as flat and dropping a 2d-gaussian on it, then smear angles
        num_photons = np.size(photon_info['energies'])
        distances = np.sqrt(-2*angle_res**2*np.log(1-np.random.random(num_photons)))
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
