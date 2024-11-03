import FermiBackgrounds
import subhalos as subs
from scipy.stats import poisson
import numpy as np
import pdb

def params_to_norm(params_in, normalize_mean, normalize_std):
    numparam = len(params_in)
    norm = (params_in - normalize_mean[0:numparam])/normalize_std[0:numparam]
    return norm

def norm_to_params(norm, normalize_mean, normalize_std):
    numparam = len(norm)
    params_out = norm*normalize_std[0:numparam] + normalize_mean[0:numparam]
    return params_out

#set up wrappers for pdf and spectra
def subhalo_pdf_wrapper(params_norm, counts, args):
    normalize_mean, normalize_std = args['normalize_mean'], args['normalize_std']
    if ('fft_type' in args):
        fft_type = args['fft_type']
    else:
        fft_type = 'fast'
    if ('physics_type' in args):
        physics_type = args['physics_type']
    else:
        physics_type = 'baseline'
    vary_DMmass = args['vary_DMmass']
    default_params = args['default_params']
    energy_range = args['energy_range']
    my_DM = args['my_DM']
    f_pf_precomputed, pf_pf_precomputed = args['f_pf_precomputed'], args['pf_pf_precomputed']
    default_phipp = args['default_phipp']
    
    params = norm_to_params(params_norm, normalize_mean, normalize_std)
    A_DM = params[0]
    if (vary_DMmass):
        mass_DM_MeV = params[2]
    else:
        mass_DM_MeV = default_params[2]

    #Spectrum of dark matter annihilation
    E = np.exp(np.linspace(np.log(energy_range[0]), np.log(energy_range[1]), 1000))
    dE = E[1:] - E[:-1]
    dNdE = my_DM.get_dNdE(E, my_DM.channel, mass_DM_MeV)

    # normalization
    PhiInt = np.sum(0.5*dE*(dNdE[1:]+dNdE[:-1]))
    sigv = 3e-26 #cm^3 s^-1 # fiducial cross section for thermal relic DM
    PhiPP = sigv*A_DM*PhiInt/(8*np.pi*(mass_DM_MeV/1e3)**2)
    PhiPP_0 = (1e-28)/(8*np.pi) #cm^3 s^-1 GeV^-2

    physics_settings, obs_settings, fft_settings = subs.get_settings(fft_type = fft_type, physics_type = physics_type)
    physics_settings['PhiPP'] = PhiPP/PhiPP_0
    pc = subs.get_pc_fast(counts, f_pf_precomputed, pf_pf_precomputed, \
                          physics_settings, fft_settings, obs_settings, \
                          default_phipp = default_phipp, do_extend = True)
    pc = pc/np.sum(pc)
    return pc

def subhalo_spec_wrapper(params_norm, energy, args):
    normalize_mean, normalize_std = args['normalize_mean'], args['normalize_std']
    params = norm_to_params(params_norm, normalize_mean, normalize_std)

    vary_DMmass = args['vary_DMmass']
    my_DM = args['my_DM']
    default_params = args['default_params']    
    
    if (vary_DMmass):
        mass_DM_MeV = params[2]
    else:
        mass_DM_MeV = default_params[2]
    return my_DM.get_dNdE(energy, my_DM.channel, mass_DM_MeV)

def background_pdf_wrapper(params_norm, counts, args):
    normalize_mean, normalize_std = args['normalize_mean'], args['normalize_std']
    params = norm_to_params(params_norm, normalize_mean, normalize_std)

    exposure = args['exposure']
    Sangle = args['Sangle']
    mean_iso_bg_flux = args['mean_iso_bg_flux']
    
    #Our model for the background is that it is Poisson distributed with
    #mean given by params[1]*mean_flux_from_fermi_isotropic_model
    mean_iso_bg = mean_iso_bg_flux*exposure*Sangle
    #Assuming Poisson background
    background_pc = poisson.pmf(counts, mean_iso_bg*params[1])    
    return background_pc

def background_spec_wrapper(params_norm, energy, args):
    normalize_mean, normalize_std = args['normalize_mean'], args['normalize_std']
    params = norm_to_params(params_norm, normalize_mean, normalize_std)

    fermi_iso = args['fermi_iso']
    
    #Using Fermi isotropic background model for spectrum
    #This is a bit weird, since this background includes non-Poisson sources (e.g. blazars)
    return fermi_iso(energy)

###########

#Distance function
def hist_dist(X,Y):
    # Y has dimension (1,nbins), since it is input data
    nbatch = X.shape[0]
    dist = np.zeros(nbatch)
    for bi in range(0,nbatch):
        diffXY = X[bi,:] - Y[0,:] 
        sumXY = X[bi,:] + Y[0,:]
        bad = np.where(sumXY == 0.)[0]
        sumXY[bad] = 1.0 #ensures that we don't get divide by zero error

        to_sum = (diffXY**2.)/sumXY
        dist[bi] = np.sqrt(np.sum(to_sum))
    return dist


def kde_dist(X,Y):
    nbatch = X.shape[0]
    nbin = X.shape[1]
    dist = np.zeros(nbatch)
    for bi in range(0,nbatch):
        tempX = gaussian_kde(X[bi,:])(np.arange(0,nbin))
        tempY = gaussian_kde(Y[0,:])(np.arange(0,nbin))
        
        fig, ax = pl.subplots(1,1)
        ax.plot(tempX)
        ax.plot(tempY)
        
        sumXY = tempX + tempY
        diffXY = tempX - tempY
        bad = np.where(sumXY == 0.)
        sumXY[bad] = 1.0 #ensures that we don't get divide by zero error

        to_sum = (diffXY**2.)/sumXY
        dist[bi] = np.sqrt(np.sum(to_sum))
    return dist

def hist_dist(X,Y):
    # Y has dimension (1,nbins), since it is input data
    nbatch = X.shape[0]
    dist = np.zeros(nbatch)
    for bi in range(0,nbatch):
        #print("X, Y = ", X.shape, Y.shape)
        #print("bi = ", bi)
        diffXY = X[bi,:] - Y[0,:] 
        sumXY = X[bi,:] + Y[0,:]
        bad = np.where(sumXY == 0.)[0]
        sumXY[bad] = 1.0 #ensures that we don't get divide by zero error

        to_sum = (diffXY**2.)/sumXY
        dist[bi] = np.sqrt(np.sum(to_sum))
    return dist

def pdist(X,Y):
    dist = np.zeros(np.size(X[:,0]))
    for i in range(np.size(dist)):
        s1 = np.sum(((X[i, np.where((X[i,:] !=0) & (Y !=0))] - Y[np.where((X[i,:] !=0) & (Y !=0))])**2)/
                    (X[i, np.where((X[i,:] !=0) & (Y !=0))] + Y[np.where((X[i,:] !=0) & (Y !=0))]))
        s2 = np.sum(((Y[np.where((X[i,:] ==0) & (Y !=0))])**2)/
                    (Y[np.where((X[i,:] ==0) & (Y !=0))]))
        s3 = np.sum(((X[i, np.where((X[i,:] !=0) & (Y ==0))])**2)/
                    (X[i, np.where((X[i,:] !=0) & (Y ==0))]))
        dist[i] = s1+s2+s3
    return dist
