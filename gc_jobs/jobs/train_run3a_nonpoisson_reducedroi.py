import numpy as np
import healpy as hp
import pickle as pk
import torch
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
from os import listdir
import sys
sys.path.append('../astroLFI')
import LFI_galactic_center
from sources import FermiBackgrounds
from sources import Model_O
from sources import DMsignal
from sources import smoothDM
from sources import MSP
from sources import Fermi_Bubbles
from sbi.inference import SNLE, SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from getdist import plots, MCSamples

parameter_range = [[], []]
abundance_luminosity_and_spectrum_list = []
source_class_list = []
parameter_names = []
energy_range = [2000, 100000] #MeV
luminosity_range = 10.0**np.array([30, 37])
max_radius = 8.5 + 20*2 #kpc
exposure = 2000*10*0.2 #cm^2 yr
#exposure = 200*10*0.2 #cm^2 yr
flux_cut = 1e-9 #photons/cm^2/s
#flux_cut = np.inf
angular_cut = 10*u.deg.to('rad') #degrees
lat_cut = 2*u.deg.to('rad') #degrees

# Add core non-poisson millisecond pulsar source clase
start1_i = np.size(parameter_range[0])
spec_file_path = '../data/MSP/1407_5583.txt'
my_MSP = MSP.MSP(spec_file_path)
def R_core_np_wrap(r, params):
    A = params[start1_i]
    #A = 0.035
    r_s = 20 #kpc
    gamma = 1.2
    return A * my_MSP.gNFW(r, r_s, gamma)**2

def Theta_core_np_wrap(theta, params):
    return np.ones(np.shape(theta))
    #return np.concatenate((np.zeros(np.round(np.size(theta)/2).astype('int')), np.ones(np.round(np.size(theta)/2).astype('int')-1)))
    #return np.sin(theta)**10

def Phi_core_np_wrap(phi, params):
    return np.ones(np.shape(phi))

def L_core_np_wrap(L, params):
    n_1 = -0.66
    n_2 = 18.2
    #L_b = params[start1_i+1] #1e34
    L_b = 1.56e37/2.35 #photons/s
    return my_MSP.luminosity_bpl(L, n_1, n_2, L_b)

def spec_core_np_wrap(energy, num_spectra, params):
    return my_MSP.MSP_spectra(energy, num_spectra)

parameter_names.append('A_{core}')
parameter_range[0].append(0.)
parameter_range[1].append(0.07)
#parameter_range[0].append(5e33)
#parameter_range[1].append(1.5e34)
MSP_core_np_als = [(R_core_np_wrap, Theta_core_np_wrap, Phi_core_np_wrap), L_core_np_wrap, spec_core_np_wrap]
abundance_luminosity_and_spectrum_list.append(MSP_core_np_als)
source_class_list.append('independent_spherical_multi_spectra')

# Add disk non-poisson millisecond pulsar source clase
start2_i = np.size(parameter_range[0])
spec_file_path = '../data/MSP/1407_5583.txt'
disk_file_path = '../data/MSP/Buschmann_etal_2020_fig7_disk.csv'
GCE_file_path = '../data/MSP/Buschmann_etal_2020_fig7_GCE.csv'
my_MSP = MSP.MSP(spec_file_path)
disk_to_GCE_source_count_ratio = my_MSP.get_disk_to_GCE_source_count_ratio(disk_file_path, GCE_file_path)
print('non-poisson disk excpected source count = ', 970*disk_to_GCE_source_count_ratio)
def R_disk_np_wrap(r, params):
    A = params[start2_i]
    #A = 0.25
    r_d = 5 #kpc
    gamma = 1.2
    return A * my_MSP.disk_R_MS(r, r_d)

def Z_disk_np_wrap(z, params):
    z_s = 0.3 #kpc
    return my_MSP.disk_Z_MS(z, z_s)

def Phi_disk_np_wrap(phi, params):
    return np.ones(np.shape(phi))

def L_disk_np_wrap(L, params):
    n_1 = -0.66
    n_2 = 18.2
    #L_b = params[1] #1e34
    L_b = 1.56e37/2.25 #photons/s
    return my_MSP.luminosity_bpl(L, n_1, n_2, L_b)

def spec_disk_np_wrap(energy, num_spectra, params):
    return my_MSP.MSP_spectra(energy, num_spectra)

parameter_names.append('A_{disk}')
parameter_range[0].append(0.)
parameter_range[1].append(2.)
MSP_disk_np_als = [(R_disk_np_wrap, Z_disk_np_wrap, Phi_disk_np_wrap), L_disk_np_wrap, spec_disk_np_wrap]
abundance_luminosity_and_spectrum_list.append(MSP_disk_np_als)
source_class_list.append('independent_cylindrical_multi_spectra')

# Add dark matter halo signal
start3_i = np.size(parameter_range[0])
N_side_DM = 2**8
N_Ebins_DM = 100
settings = {'N_side': N_side_DM, 'theta_cutoff': angular_cut, 'halo_dist': 8.5*u.kpc.to('cm'), 'Rs': 20*u.kpc.to('cm'), 'mass_func': 'gNFW'}
my_DM_flux = smoothDM.smoothDM(**settings)
channel = 'b'
DM_directory = '../data/dm_spectra/'
my_DM_signal = DMsignal.DMsignal(DM_directory, channel)

def DM_wrap(params):
    DM_mass = params[start3_i] #MeV
    cross_sec = params[start3_i + 1]*1e-26 #cm^3 s^-1
    rho_s = 1.06e-2*u.Msun.to('kg')*c.c.value**2*u.J.to('MeV')/u.pc.to('cm')**3
    r_s = 20*u.kpc.to('cm')
    gamma = 1.2
    DM_energies = np.geomspace(energy_range[0], energy_range[1], N_Ebins_DM+1) #MeV
    dNdE = my_DM_signal.get_dNdE(DM_energies, channel, DM_mass)
    mass_func_params = (r_s, rho_s, gamma)
    DM_map, DM_indices = my_DM_flux.get_map(DM_mass, cross_sec, dNdE, mass_func_params)
    return DM_map.T, DM_energies, DM_indices, N_side_DM

parameter_names.append(r'm_{\chi}')
parameter_range[0].append(10000.)
parameter_range[1].append(100000.)
parameter_names.append(r'\langle\sigma_{DM} v\rangle_0')
parameter_range[0].append(0.)
parameter_range[1].append(10.0)
DM_als = [DM_wrap]
abundance_luminosity_and_spectrum_list.append(DM_als)
source_class_list.append('healpix_map')

#sbi
def simulator(params):
    params = params.numpy()
    N_side = 2**6
    N_Ebins = 10
    source_info = my_LFI.create_sources(params, grains = 4000)
    photon_info = my_LFI.generate_photons_from_sources(params, source_info, grains = 1000)
    obs_info = {'psf_fits_path': '../paper2/paper2_data/Fermi_files/psf_P8R3_ULTRACLEANVETO_V2_PSF.fits', 'edisp_fits_path': '../paper2/paper2_data/Fermi_files/edisp_P8R3_ULTRACLEANVETO_V2_PSF.fits', 'event_type': 'PSF3', 'exposure_map': None}
    obs_photon_info = my_LFI.mock_observe(photon_info, obs_info)
    heatmap = my_LFI.get_partial_map_summary(obs_photon_info, N_side, N_Ebins)
    #print(heatmap[0].shape)
    return heatmap[0].flatten()
prior = utils.BoxUniform(low = parameter_range[0], high = parameter_range[1])

load_posterior = False
load_sims = True
posterior_file = 'posteriors/run3a_nonpoisson_reducedroi/1000000sims'
sim_dir = 'simulations/run3a_nonpoisson_reducedroi/'
if load_posterior:
    posterior = np.load(posterior_file + '.npy', allow_pickle=True)[()]
else:
    if load_sims:
        sims = listdir(sim_dir)
        theta, x = torch.tensor([]), torch.tensor([])
        i = 0
        for sim in sims:
            i += 1
            theta_batch, x_batch = np.load(sim_dir + sim, allow_pickle=True)
            theta = torch.cat((theta, theta_batch))
            x = torch.cat((x, x_batch))
            print(i,sim)
            if i == 1000:
                break
    else:
        simulator, prior = prepare_for_sbi(simulator, prior)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=10)
    inference = SNPE(prior=prior)
    #inference = SNPE(prior=prior, device='cuda')
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    np.save(posterior_file, posterior, allow_pickle=True)