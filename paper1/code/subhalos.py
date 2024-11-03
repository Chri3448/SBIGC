import numpy as np
from astropy import units
from math import e
import pdb
import matplotlib.pyplot as pl

if (0):
    num_M = 200
    num_ell = 400
if (1):
    num_M = 400
    num_ell = 800

def mass_function(settings, mass, gc_distance = 8.5):
    #settings should specify
    #M_min
    #M_max
    #beta
    
    M_min = settings['M_min']
    M_max = settings['M_max']
    beta = settings['beta']

    normalization = (1.2e4) #1/Msun/kpc^3
    #scale radius of MW
    r_s = 21. #kpc
    mass_func_ret = normalization*(mass**(-beta))/((gc_distance/r_s)*(1.+ gc_distance/r_s)**2)
    above = np.where(mass > M_max)[0]
    below = np.where(mass < M_min)[0]
    mass_func_ret[above] = 0.
    mass_func_ret[below] = 0.
    
    return mass_func_ret

def get_meansigma_lnL(settings, mass, gc_distance = 8.5):
    #lum in photons/sec
    #mass in Msun
    #rad in kpc

    #settings should specify
    #PhiPP
    #n
    #lum_model
    PhiPP = settings['PhiPP']
    n = settings['n']
    if (not 'lum_model' in settings):
        lum_model = 'default'
    else:
        lum_model = settings['lum_model']

    # C0 model from Koushiappas et al.
    if (lum_model == 'default'):
        nterm = (n/2.)*(0.62*np.log(mass/1.0e5) - 0.08*np.log(gc_distance/50.))
        mean_lnL = 77.4 + 0.87*np.log(mass/1.0e5) - 0.23*np.log(gc_distance/50.) + np.log(PhiPP) + nterm
        sigma_lnL = 0.74 - 0.003*np.log(mass/1.0e5) + 0.011*np.log(gc_distance/50.)
        
    if (lum_model == 'C+'):
        mean_lnL = 77.5 + 0.87*np.log(mass/1.0e5) - 0.26*np.log(gc_distance/50.) + np.log(PhiPP)
        sigma_lnL = 0.76 - 0.0021*np.log(mass/1.0e5) + 0.0077*np.log(gc_distance/50.)
    if (lum_model == 'C-'):
        mean_lnL = 77.3 + 0.87*np.log(mass/1.0e5) - 0.18*np.log(gc_distance/50.) + np.log(PhiPP)
        sigma_lnL = 0.75 - 0.0013*np.log(mass/1.0e5) + 0.0044*np.log(gc_distance/50.)

    if (lum_model == 'C0-simplified'):
        nterm = (n/2.)*(0.62*np.log(mass/1.0e5) - 0.08*np.log(gc_distance/50.))
        mean_lnL = 77.4 + 0.87*np.log(mass/1.0e5) - 0.23*np.log(gc_distance/50.) + np.log(PhiPP) + nterm
        sigma_lnL = 0.74 + 0.011*np.log(gc_distance/50.)

    return mean_lnL, sigma_lnL
    

def P_of_L(settings, luminosity, mass, gc_distance = 8.5):
    mean_lnL, sigma_lnL = get_meansigma_lnL(settings, mass, gc_distance = gc_distance)
    
    lnL = np.log(luminosity)
    P_of_lnL = (1./(sigma_lnL*np.sqrt(2.*np.pi) ))*np.exp(-0.5*((lnL - mean_lnL)/sigma_lnL**2.)**2.)
    return P_of_lnL/luminosity #1/L = s    

def l_integrand(ell, F, settings, obs_settings):
    psi = obs_settings['psi']
    M_min = settings['M_min']
    M_max = settings['M_max']
    n = settings['n']
    beta = settings['beta']
    PhiPP = settings['PhiPP']
    d_sun = settings['d_sun']
    R_G = settings['R_G']
    
    ell = np.atleast_1d(ell)
    numl = len(ell)
    rad = np.sqrt(ell**2. + d_sun**2. - 2.*ell*d_sun*np.cos(psi))
    lum_desired = 4.*np.pi*(ell**2.)*F

    lintegrand_ret = np.zeros(numl)
    for li in range(numl):
        L_cent = 4.*np.pi*F*ell[li]**2.
        M_cent = 1.0e5*np.exp((np.log(L_cent) - 77.4 + 0.23*np.log(rad[li]/50.) - np.log(PhiPP))/0.87)
        M_min_integrate = 0.0005*M_cent #M_sun
        M_max_integrate = 5000.*M_cent #M_sun
        if ((M_max_integrate > M_min) and (M_min_integrate < M_max)):
            MM = np.exp(np.linspace(np.log(M_min_integrate), np.log(M_max_integrate), num = num_M))
            dM = MM[1:] - MM[:-1]

            mass_func_vals = mass_function(settings, MM, gc_distance = rad[li]) #1./Msun*kpc^3
            P_of_L_vals = P_of_L(settings, lum_desired[li], MM, gc_distance = rad[li]) # s
            integrand_vals = mass_func_vals*P_of_L_vals #s/Msun*kpc^3
            integral_result = np.sum(0.5*dM*(integrand_vals[1:] + integrand_vals[:-1])) #s/kpc^3
            lintegrand_ret[li] = (ell[li]**4.)*integral_result #s kpc
    return lintegrand_ret #kpc

def p_1_f(F_cm2_yr, settings, obs_settings):
    psi = obs_settings['psi']
    M_min = settings['M_min']
    M_max = settings['M_max']
    n = settings['n']
    beta = settings['beta']
    PhiPP = settings['PhiPP']
    d_sun = settings['d_sun']
    R_G = settings['R_G']
    l_min = settings['l_min']
    
    F = np.atleast_1d(F_cm2_yr)
    F_kpc2_s = F*((units.kpc/units.cm).to('')**2.)*(units.s/units.yr).to('')
    l_max = d_sun*(np.cos(psi)  + np.sqrt(-np.sin(psi)**2. + (R_G/d_sun)**2.)) 
    p_1_f_ret = np.zeros(len(F))

    for fi in range(0, len(F)):
              
        ll = np.exp(np.linspace(np.log(l_min), np.log(l_max), num = num_ell))
        dl = ll[1:] - ll[:-1]
        integrand_vals = l_integrand(ll, F_kpc2_s[fi], settings, obs_settings) #s kpc
        integral_result = np.sum(0.5*dl*(integrand_vals[1:] + integrand_vals[:-1])) #s kpc^2
        p_1_f_ret[fi] = integral_result #1/F so s kpc^2
        #pdb.set_trace()
    
    #units are 1/F so yr*cm^2
    return p_1_f_ret


def get_mu(settings, obs_settings):
    Omega_pixel = obs_settings['Omega_pixel']
    psi = obs_settings['psi']
    M_min = settings['M_min']
    M_max = settings['M_max']
    beta = settings['beta']
    d_sun = settings['d_sun']
    R_G = settings['R_G']
    l_min = settings['l_min']
    l_max = d_sun*(np.cos(psi)  + np.sqrt(-np.sin(psi)**2. + (R_G/d_sun)**2.)) 
    ll = np.exp(np.linspace(np.log(l_min), np.log(l_max), num = 400))
    rad = np.sqrt(ll**2. + d_sun**2. - 2.*ll*d_sun*np.cos(psi))
    dl = ll[1:] - ll[:-1]
    l_integrand = np.zeros(len(ll))
    for li in range(0, len(ll)):
        M_min_integrate = M_min #M_sun
        M_max_integrate = M_max #M_sun
        MM = np.exp(np.linspace(np.log(M_min_integrate), np.log(M_max_integrate), num = num_M))
        dM = MM[1:] - MM[:-1]
        dNdM = mass_function(settings, MM, rad[li]) #1/Msun/kpc^3
        l_integrand[li] = (ll[li]**2.)*np.sum(0.5*dM*(dNdM[1:] + dNdM[:-1])) #1/kpc
    mu = Omega_pixel*np.sum(0.5*dl*(l_integrand[1:] + l_integrand[:-1]))
    return mu

def Pf_to_Pc(ff, Pf, exposure, C_array, norm = 1.0, minf = 0.0, maxf = np.inf, extend_powerlaw = {'do_extend':False, 'extend_law':0, 'new_maxf':0.0}):
    goodf = np.where((ff > minf) & (ff < maxf))[0]
    ff_restrict = ff[goodf]
    Pf_restrict = Pf[goodf]
    
    #Scale for normalization
    ff_restrict = ff_restrict*norm
    Pf_restrict = Pf_restrict/norm

    #extend P(F) with a powerlaw
    if (extend_powerlaw['do_extend']):
        new_maxf = extend_powerlaw['new_maxf']
        extra_ff = np.exp(np.linspace(np.log(np.max(ff_restrict)), np.log(new_maxf), num = 100))
        extra_Pf = Pf_restrict[-1]*(extra_ff/extra_ff[0])**extend_powerlaw['extend_law']
        new_ff_restrict = np.append(ff_restrict, extra_ff[1:])
        new_Pf_restrict = np.append(Pf_restrict, extra_Pf[1:])
        ff_restrict = new_ff_restrict
        Pf_restrict = new_Pf_restrict

        #fig, ax = pl.subplots(1,1)
        #ax.plot(ff_restrict, Pf_restrict)
        #ax.plot(extra_ff, extra_Pf)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        
    
    df = ff_restrict[1:] - ff_restrict[:-1]
    #print("pf integral = ", np.sum(0.5*df*(Pf_restrict[1:] + Pf_restrict[:-1])))
    
    Pc = np.zeros(len(C_array))
    for ci in range(0,len(C_array)):
        #if (C_array[ci] < 12):
        #    lnfac = np.log(np.math.factorial(C_array[ci]))
        #else:
        #    lnfac = C_array[ci]*np.log(C_array[ci]) - C_array[ci]
        lnfac = np.math.lgamma(C_array[ci] + 1) # log gamma (n) = log(factorial(n+1))
        lnpoisson = C_array[ci]*np.log(exposure*ff_restrict) - exposure*ff_restrict - lnfac
        to_integrate = Pf_restrict*np.exp(lnpoisson)
        Pc[ci] = np.sum(0.5*df*(to_integrate[1:] + to_integrate[:-1]))

    return Pc

def get_pc(cc, physics_settings, fft_settings, obs_settings, return_fpf = False):
    #photons/yr/cm^2
    minF = fft_settings['minF']
    maxF = fft_settings['maxF']
    numF = fft_settings['numF']
    F = np.exp(np.linspace(np.log(minF), np.log(maxF), num = numF))
    dF = F[1:] - F[:-1]

    #truncating pf when calculating P(C) to get rid of oscillations
    minf_pc = fft_settings['minf_pc']
    maxf_pc = fft_settings['maxf_pc']

    #get mu
    mu = get_mu(physics_settings, obs_settings)
    print("mu = ", mu)

    #Get P_1(F)
    p1f = p_1_f(F, physics_settings, obs_settings)
    p1f_normalized = p1f/np.sum(0.5*dF*(p1f[1:] + p1f[:-1]))
    
    #Get P_sh(F)
    f_pf, pf_pf = P1f_to_Pf(F, p1f_normalized, mu, fft_settings)

    #Get P(C)
    fov_factor = obs_settings['fov_factor']
    obs_time = obs_settings['obs_time']
    area = obs_settings['area']
    exposure = fov_factor*(area*obs_time)
    Pc = Pf_to_Pc(f_pf, pf_pf, exposure, cc, norm = 1.0,  minf = minf_pc, maxf = maxf_pc)
    if (not return_fpf):
        return Pc
    else:
        return Pc, f_pf, pf_pf

def get_pc_fast(cc, f_pf_precomputed, pf_pf_precomputed, physics_settings, fft_settings, obs_settings, default_phipp = 1.0, do_extend = False):
    norm = physics_settings['PhiPP']/default_phipp
    fov_factor = obs_settings['fov_factor']
    obs_time = obs_settings['obs_time']
    area = obs_settings['area']
    exposure = fov_factor*area*obs_time

    #truncating pf when calculating P(C) to get rid of oscillations
    minf_pc = fft_settings['minf_pc']
    maxf_pc = fft_settings['maxf_pc']

    gamma = -1.03/(1.+0.36*physics_settings['n']) - 1.
    # hardcoding factor of 1000 here...not great
    extend_powerlaw = {'do_extend':do_extend, 'extend_law':gamma, 'new_maxf':1000*maxf_pc}
    
    Pc = Pf_to_Pc(f_pf_precomputed, pf_pf_precomputed, exposure, cc, norm = norm,  minf = minf_pc, maxf = maxf_pc, extend_powerlaw = extend_powerlaw)

    return Pc
    
def P1f_to_Pf(f_input, p1f, mu, fft_settings):
    f_new = np.copy(f_input)
    p1f_new = np.copy(p1f)

    #fig, ax  = pl.subplots(1,1)
    #ax.plot(f_new, p1f_new)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    minlnk = fft_settings['minlnk']
    maxlnk = fft_settings['maxlnk']
    numk = fft_settings['numk']

    minf_out = fft_settings['minf_out']
    maxf_out = fft_settings['maxf_out']
    numf_out = fft_settings['numf_out']
    
    lnk = np.linspace(minlnk, maxlnk, num = numk)
    kk = np.exp(lnk)
    dlnk = lnk[1:] - lnk[:-1]
    
    p1k = logfourier_fast(np.log(f_new), p1f_new, kk, -1.0)
        
    #temp = np.exp(mu*(p1k - 1.))
    temp = e**(mu*(p1k - 1.))


    #fig, ax = pl.subplots(1,1)
    #ax.plot(kk, np.real(temp))
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    f_output = np.exp(np.linspace(np.log(minf_out), np.log(maxf_out), num = numf_out))
    
    pf = logfourier_fast(np.log(kk), temp, f_output, 1.0)
    pf_output = np.real((pf + pf.conjugate()))
     
    return f_output, pf_output    
    
def logfourier_fast(uu, ff, kk, direction):
    #uu = ln(x), where x are evenly spaced
    #ff = function(uu)
    #kk = desired k values
    #returns fourier transform evaluated at kk
    #print("direction = ", direction)
    
    npts = len(ff)
    du = uu[1:] - uu[:-1]
    
    xx = np.exp(uu)
    dx = xx[1:] - xx[:-1]
    numk = len(kk)
    fk = np.zeros(numk, dtype=complex)
    for ki in range(0,numk):
        #temp = ff*np.exp(direction*1j*kk[ki]*xx)
        temp = ff*e**(2.*np.pi*direction*1j*kk[ki]*xx)
        fk[ki] = (np.sum(0.5*dx*(temp[1:] + temp[:-1])))
        
    return fk
    

def get_gamma(n, beta):
    alpha = 0.87+0.31*n
    gamma = (1.0-beta)/alpha - 1.0
    return gamma

def test():
    physics_settings, obs_settings, fft_settings = get_settings(fft_type = 'fast')
    counts = np.arange(0,20)
    pc = get_pc(counts, physics_settings, fft_settings_fast, obs_settings)
    print("pc = ", pc)

def get_settings(fft_type = 'fast', physics_type = 'baseline'):
    if (fft_type == 'fast_old'):
        fft_settings = {
            'minF':1.0e-16,
            'maxF':1.0e5,
            'numF':2**8,
            'minlnk':np.log(1.0e1),
            'maxlnk':np.log(1.0e10),
            'numk':2**8,
            'minf_out':1.0e-9,
            'maxf_out':1.0e-4,
            'numf_out':2**8
        }
    if (fft_type == 'fast'):
        fft_settings = {
            'minF':1.0e-16,
            'maxF':1.0e5,
            'numF':2**13,
            'minlnk':np.log(1.0e-3),#np.log(1.0e1),
            'maxlnk':np.log(5.0e6),#np.log(1.0e10),
            'numk':2**13,
            'minf_out':1e-5,#7.0e-5,
            'maxf_out':10.0e-4,#10.0e-4,
            'numf_out':2**9,
            'minf_pc':1.0e-5,  #how to restrict when computing P(C)
            'maxf_pc':8.0e-4
        }
    if (fft_type == 'fast_isotropic'):
        fft_settings = {
            'minF':1.0e-16,
            'maxF':1.0e5,
            'numF':2**13, #13
            'minlnk':np.log(1.0e-3),#np.log(1.0e1),
            'maxlnk':np.log(5.0e6),#np.log(1.0e10),
            'numk':2**13, #13
            'minf_out':7.0e-5,
            'maxf_out':5.0e-3, #2.0e-3,
            'numf_out':2**9,
            'minf_pc':7.0e-5,  #how to restrict when computing P(C)
            'maxf_pc':5.0e-3   #2.0e-3
        }
    if (fft_type == 'highMmin'):
        fft_settings = {
            'minF':1.0e-16,
            'maxF':1.0e5,
            'numF':2**13, #13
            'minlnk':np.log(1.0e-3),#np.log(1.0e1),
            'maxlnk':np.log(5.0e6),#np.log(1.0e10),
            'numk':2**13, #13
            'minf_out':1e-7,#7.0e-5,
            'maxf_out':10.0e-4,#10.0e-4,
            'numf_out':2**9,
            'minf_pc':3.0e-7,  #how to restrict when computing P(C)
            'maxf_pc':3.0e-4
        }
    if (fft_type == 'highacc'):
        fft_settings= {
            'minF':1.0e-16,
            'maxF':1.0e5,
            'numF':2**11,
            'minlnk':np.log(1.0e1),
            'maxlnk':np.log(1.0e10),
            'numk':2**11,
            'minf_out':1.0e-9,
            'maxf_out':1.0e-4,
            'numf_out':2**11
        }
    obs_settings = {
        'area':2000.0, #cm^2
        'obs_time':10.0, #yr
        'fov_factor':0.2, #((1- (np.sqrt(3)/2))/2)
        'Omega_pixel':4.*np.pi/(12*64**2.), #sr
        'psi':100.*np.pi/180.        
    }
    if (physics_type == 'baseline'):
        physics_settings = {
            'R_G':220,
            'd_sun':8.5, 
            'M_max':1.0e10,
            'M_min':1.0,
            'PhiPP':1.0,
            'n':0,
            'beta':1.9,
            'lum_model':'default',
            'l_min':0.001
        }
    if (physics_type == 'isotropic'):
        physics_settings = {
            'R_G':220,
            'd_sun':0.001, 
            'M_max':1.0e10,
            'M_min':1.0,
            'PhiPP':1.0,
            'n':0,
            'beta':1.9,
            'lum_model':'C0-simplified',
            'l_min':0.001
        }
    return physics_settings, obs_settings, fft_settings
    
    
#functions for quick analysis
def mock_spectrum(param_dict, energy):
    #define a spectrum here as a function of parameters
    #doesn't need to be normalized
    powerlaw = param_dict['power_law']
    return energy**powerlaw


def pdf_source1(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    #Poisson PDF with mean specified by params[0]
    return poisson.pmf(counts, params[0])

def pdf_source2(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    #doesn't need to be normalized
    return (counts**params[1])*np.exp(-(counts/50.)**2.)

def spectrum_source1(params, energy):
    #define a spectrum here as a function of parameters
    #doesn't need to be normalized
    return energy**params[2]

def spectrum_source2(params, energy):
    #define a spectrum here as a function of parameters
    return energy**(params[3])

def lnprior(params):
    return 0.

    
if __name__ == "__main__":
    test()
