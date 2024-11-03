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
    
   
