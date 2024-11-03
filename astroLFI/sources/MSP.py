import numpy as np
import scipy as sp
from sklearn.neighbors import KernelDensity
'''
Class with functions for luminosity distributions, spatial distributions, and spectra of millisecond pulsars based on Dinsmore et al. (2021)
'''
class MSP:
    
    def __init__(self, file_path):
        self.Ecut_vals, self.alpha_vals, self.Ecut_alpha_PDF = self.construct_Ecut_alpha_PDF(file_path, 1000)
    
    '''
    Luminosity distribution functions
    '''
    def luminosity_PL(self, L, alpha, Lmin, L_max):
        prefac = L**-alpha*np.exp(-L/L_max)
        P_PL = prefac / sp.special.gamma(1-alpha)*sp.special.gammaincc(1-alpha, L_min/L_max)*L_max**(1-alpha)
        
        return P_PL
    
    def luminosity_LN(self, L, L_0, sig):
        prefac = np.log10(e)/(sig*np.sqrt(2*np.pi)*L)
        P_LN = prefac*np.exp(-(np.log10(L) - np.log10(L_0))**2/(2*sig**2))
        
        return P_LN
        
    def luminosity_bpl(self, L, n_1, n_2, L_b):
        prefac = (1-n_1)*(1-n_2)/L_b/(n_1-n_2)
        P_BPL = prefac*np.concatenate(((L[np.where(L<=L_b)]/L_b)**-n_1, (L[np.where(L>L_b)]/L_b)**-n_2))
        
        return P_BPL
    
    '''
    Spacial distribution functions
    '''
    def gNFW(self, r, r_s, gamma):
        
        return (r/r_s)**-gamma * (1+(r/r_s))**(-3+gamma)
    
    #ploeg models from arXiv:2008.10821v4
    def disk_R_ploeg(self, r, sig_r):
        
        return A * np.exp(-r**2/(2*sig_r**2)) * np.exp(-np.abs(z)/z_0)
    
    def disk_Z_ploeg(self, z, z_0):
        
        return A * np.exp(-r**2/(2*sig_r**2)) * np.exp(-np.abs(z)/z_0)
    
    #MS models from arXiv:2110.06931v2 
    def disk_R_MS(self, r, r_d):
        
        return np.exp(-r/(r_d))

    def disk_Z_MS(self, z, z_s):
        
        return np.exp(-np.abs(z)/z_s)
    
    '''
    Spacial normalization functions
    '''
    def get_disk_to_GCE_source_count_ratio(self, disk_file_path, GCE_file_path, grains = 10000):
        disk_F = np.loadtxt(disk_file_path, delimiter=',', skiprows = 1)[:,0]
        disk_F2dNdF = np.loadtxt(disk_file_path, delimiter=',', skiprows = 1)[:,1]
        GCE_F = np.loadtxt(GCE_file_path, delimiter=',', skiprows = 1)[:,0]
        GCE_F2dNdF = np.loadtxt(GCE_file_path, delimiter=',', skiprows = 1)[:,1]
        disk_fit = sp.interpolate.interp1d(np.log(disk_F), np.log(disk_F2dNdF), kind  = 'linear')
        GCE_fit = sp.interpolate.interp1d(np.log(GCE_F), np.log(GCE_F2dNdF), kind  = 'linear')
        disk_F = np.exp(np.linspace(np.log(disk_F[0]), np.log(disk_F[-1]), grains))
        GCE_F = np.exp(np.linspace(np.log(GCE_F[0]), np.log(GCE_F[-1]), grains))
        disk_dNdF = np.exp(disk_fit(np.log(disk_F)))/disk_F**2
        GCE_dNdF = np.exp(GCE_fit(np.log(GCE_F)))/GCE_F**2
        disk_dF = disk_F[1:] - disk_F[:-1]
        GCE_dF = GCE_F[1:] - GCE_F[:-1]
        disk_norm = np.sum(disk_dNdF[:-1]*disk_dF)
        GCE_norm = np.sum(GCE_dNdF[:-1]*GCE_dF)
        
        return disk_norm/GCE_norm
    
    '''
    Spectra functions
    '''
    def construct_Ecut_alpha_PDF(self, file_path, grains):
        #returns PDF for Ecut (in MeV) and alpha from arXiv:1407.5583
        MSP_file = file_path
        MSP_data = np.genfromtxt(MSP_file, names = True)
        X = np.concatenate((MSP_data['Ecut'][:, np.newaxis], MSP_data['alpha'][:, np.newaxis]), axis=1)
        kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(X)
        X_vals = np.linspace(0.01,10,grains)
        Y_vals = np.linspace(-3,1,grains)
        Xm, Ym = np.meshgrid(X_vals, Y_vals)
        coordinates = np.concatenate((Xm.flatten()[:, np.newaxis], Ym.flatten()[:, np.newaxis]), axis=1)
        log_dens = kde.score_samples(coordinates)
        
        return X_vals*1000, Y_vals, np.exp(log_dens.reshape((grains, grains))).T #should be transposed, when ready, remove # and retrain all networks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    def MSP_spectra(self, energy, num_spectra = 1, fixed_spectra = False, Ecut = np.nan, alpha = np.nan):
        #returns num_spectra realizations of a MSP spectrum drawn from a parameter distribution
        #if fixed spectrum is True, but either Ecut or alpha are not specififed, the value that maximixes the pdf is returned
        #for energy given in MeV
        if fixed_spectra:
            if np.logical_or(np.isnan(Ecut), np.isnan(alpha)):
                max_i = np.unravel_index(np.argmax(self.Ecut_alpha_PDF), self.Ecut_alpha_PDF.shape)
                Ecut_i, alpha_i = max_i[0]*np.ones(num_spectra, dtype = 'int'), max_i[1]*np.ones(num_spectra, dtype = 'int')
                Ecut = np.tile(self.Ecut_vals[Ecut_i], (np.size(energy), 1)).T
                alpha = np.tile(self.alpha_vals[alpha_i], (np.size(energy), 1)).T
            else:
                Ecut = np.tile(Ecut*np.ones(num_spectra), (np.size(energy), 1)).T
                alpha = np.tile(alpha*np.ones(num_spectra), (np.size(energy), 1)).T
        else:
            Ecut_i, alpha_i = self.draw_from_2D_pdf(self.Ecut_alpha_PDF, Ndraws = num_spectra)
            Ecut = np.tile(self.Ecut_vals[Ecut_i], (np.size(energy), 1)).T
            alpha = np.tile(self.alpha_vals[alpha_i], (np.size(energy), 1)).T
        energy_m = np.tile(energy, (num_spectra, 1))
        spec = (energy_m/Ecut)**alpha*np.exp(-energy_m/Ecut)
        norms = np.sum(spec[:,:-1]*(energy_m[:,1:]-energy_m[:,:-1]), axis = 1)
        
        return spec/(np.tile(norms, (np.size(energy), 1)).T)
    
    def MSP_spectra_load(self, energy, file, num_spectra = 1):
        #returns num_spectra copies of loaded spectrum evaluated at energy
        data = np.load(file)
        loaded_energy, loaded_spec = data[0], data[1]
        spec_func = sp.interpolate.interp1d(loaded_energy, loaded_spec, fill_value = 'extrapolate')
        energy_m = np.tile(energy, (num_spectra, 1))
        spec =  np.tile(spec_func(energy), (num_spectra, 1))
        norms = np.sum(spec[:,:-1]*(energy_m[:,1:]-energy_m[:,:-1]), axis = 1)
        
        return spec/(np.tile(norms, (np.size(energy), 1)).T)
    
        
    '''
    Statistcal funstions
    '''
    #Takes a 2D array. pdf[x,y] should equal pdf(x,y). Returns two 1D arrays of x and y indices.
    #pdf*dx*dy must be passed to this func as pdf for normalization. If x or y are not linspaced, the pdf dimensions should be (n-1,m-1),
    #and the last indices of x and y will not be drawn
    def draw_from_2D_pdf(self, pdf, Ndraws = np.nan):
        if Ndraws == np.nan:
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