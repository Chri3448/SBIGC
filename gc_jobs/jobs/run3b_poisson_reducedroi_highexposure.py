import numpy as np
import healpy as hp
import pickle as pk
import torch
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
import sys
import os
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
    
def run_sims(num_sims):
    
    parameter_range = [[], []]
    abundance_luminosity_and_spectrum_list = []
    source_class_list = []
    parameter_names = []
    energy_range = [2000, 100000] #MeV
    luminosity_range = 10.0**np.array([30, 37])
    max_radius = 8.5 + 20*2 #kpc
    exposure = 10*2000*10*0.2 #cm^2 yr
    #exposure = 200*10*0.2 #cm^2 yr
    flux_cut = 1e-9 #photons/cm^2/s
    #flux_cut = np.inf
    angular_cut = 10*u.deg.to('rad') #degrees
    lat_cut = 2*u.deg.to('rad') #degrees
    
    # Add core poisson millisecond pulsar source clase
    start1_i = np.size(parameter_range[0])
    spec_file_path = '../data/MSP/1407_5583.txt'
    my_MSP = MSP.MSP(spec_file_path)
    def R_core_p_wrap(r, params):
        A = params[start1_i]
        #A = 1.0
        r_s = 20 #kpc
        gamma = 1.2
        return A * my_MSP.gNFW(r, r_s, gamma)**2

    def Theta_core_p_wrap(theta, params):
        return np.ones(np.shape(theta))
        #return np.concatenate((np.zeros(np.round(np.size(theta)/2).astype('int')), np.ones(np.round(np.size(theta)/2).astype('int')-1)))
        #return np.sin(theta)**10

    def Phi_core_p_wrap(phi, params):
        return np.ones(np.shape(phi))

    def L_core_p_wrap(L, params):
        n_1 = 0.97
        n_2 = 2.6
        #L_b = params[start1_i+1] #1e34
        L_b = 1.061e36/2.25 #photons/s
        return my_MSP.luminosity_bpl(L, n_1, n_2, L_b)

    def spec_core_p_wrap(energy, num_spectra, params):
        return my_MSP.MSP_spectra(energy, num_spectra)

    parameter_names.append('A_{core}')
    parameter_range[0].append(0.)
    parameter_range[1].append(3.)
    #parameter_range[0].append(5e33)
    #parameter_range[1].append(1.5e34)
    MSP_core_p_als = [(R_core_p_wrap, Theta_core_p_wrap, Phi_core_p_wrap), L_core_p_wrap, spec_core_p_wrap]
    abundance_luminosity_and_spectrum_list.append(MSP_core_p_als)
    source_class_list.append('independent_spherical_multi_spectra')
    
    # Add disk poisson millisecond pulsar source clase
    start2_i = np.size(parameter_range[0])
    spec_file_path = '../data/MSP/1407_5583.txt'
    disk_file_path = '../data/MSP/Buschmann_etal_2020_fig7_disk.csv'
    GCE_file_path = '../data/MSP/Buschmann_etal_2020_fig7_GCE.csv'
    my_MSP = MSP.MSP(spec_file_path)
    disk_to_GCE_source_count_ratio = my_MSP.get_disk_to_GCE_source_count_ratio(disk_file_path, GCE_file_path)
    print('poisson disk excpected source count = ', 2.6e4*disk_to_GCE_source_count_ratio)
    def R_disk_p_wrap(r, params):
        A = params[start2_i]
        #A = 5.0
        r_d = 5 #kpc
        gamma = 1.2
        return A * my_MSP.disk_R_MS(r, r_d)

    def Z_disk_p_wrap(z, params):
        z_s = 0.3 #kpc
        return my_MSP.disk_Z_MS(z, z_s)

    def Phi_disk_p_wrap(phi, params):
        return np.ones(np.shape(phi))

    def L_disk_p_wrap(L, params):
        n_1 = 0.97
        n_2 = 2.60
        #L_b = params[1] #1e34
        L_b = 1.061e36/2.25 #photons/s
        return my_MSP.luminosity_bpl(L, n_1, n_2, L_b)

    def spec_disk_p_wrap(energy, num_spectra, params):
        return my_MSP.MSP_spectra(energy, num_spectra)

    parameter_names.append('A_{disk}')
    parameter_range[0].append(0.)
    parameter_range[1].append(15.)
    MSP_disk_p_als = [(R_disk_p_wrap, Z_disk_p_wrap, Phi_disk_p_wrap), L_disk_p_wrap, spec_disk_p_wrap]
    abundance_luminosity_and_spectrum_list.append(MSP_disk_p_als)
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
    parameter_range[1].append(100000)
    parameter_names.append(r'\langle\sigma_{DM} v\rangle_0')
    parameter_range[0].append(0.)
    parameter_range[1].append(10.0)
    DM_als = [DM_wrap]
    abundance_luminosity_and_spectrum_list.append(DM_als)
    source_class_list.append('healpix_map')
    
    # Add Fermi isotropic background source class
    my_FB = FermiBackgrounds.FermiBackgrounds('..')
    def spec_iso_wrap(energy, params):
        iso_fit = my_FB.get_isotropic_background_spectrum_func()
        return iso_fit(energy)
    FIB_als = [spec_iso_wrap]
    abundance_luminosity_and_spectrum_list.append(FIB_als)
    source_class_list.append('isotropic_diffuse')
    
    # Add Model O source class
    my_MO = Model_O.Model_O('..')
    N_side_MO = 2**8
    N_Ebins = 1500

    # inverse compton scattering model:
    MO_ics_map, MO_ics_energies, MO_ics_indices = my_MO.get_partial_map_ics(angular_cut, energy_range[0], energy_range[1], N_Ebins, N_side_MO)
    def MO_ics_wrap(params):
        return MO_ics_map, MO_ics_energies, MO_ics_indices, N_side_MO

    MO_ics_als = [MO_ics_wrap]
    abundance_luminosity_and_spectrum_list.append(MO_ics_als)
    source_class_list.append('healpix_map')

    # pi^0 + Bremsstrahlung model:
    MO_pibrem_map, MO_pibrem_energies, MO_pibrem_indices = my_MO.get_partial_map_pibrem(angular_cut, energy_range[0], energy_range[1], N_Ebins, N_side_MO)
    def MO_pibrem_wrap(params):
        return MO_pibrem_map, MO_pibrem_energies, MO_pibrem_indices, N_side_MO

    MO_pibrem_als = [MO_pibrem_wrap]
    abundance_luminosity_and_spectrum_list.append(MO_pibrem_als)
    source_class_list.append('healpix_map')
    
    # Add Fermi Bubbles source class
    my_FBub = Fermi_Bubbles.Fermi_Bubbles('..')
    N_side_FBub = 2**8
    N_Ebins = 1500

    FBub_map, FBub_energies, FBub_indices = my_FBub.get_partial_map(angular_cut, energy_range[0], energy_range[1], N_Ebins, N_side_FBub)
    def FBub_wrap(params):
        return FBub_map, FBub_energies, FBub_indices, N_side_FBub

    FBub_als = [FBub_wrap]
    abundance_luminosity_and_spectrum_list.append(FBub_als)
    source_class_list.append('healpix_map')
    
    my_LFI = LFI_galactic_center.LFI_G(abundance_luminosity_and_spectrum_list, source_class_list, parameter_range, energy_range, luminosity_range, max_radius, exposure, angular_cut, lat_cut, flux_cut, verbose = False)

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
    simulator, prior = prepare_for_sbi(simulator, prior)
    print(num_sims)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sims)
    
    return theta, x

if (__name__=="__main__"):
    job_number = sys.argv[1]
    path = '/home/chri3448/EPDF_ABC/gc_jobs/simulations/run3b_poisson_reducedroi_highexposure'
    theta, x = run_sims(num_sims = 100)
    os.makedirs(path, exist_ok=True)
    np.save(path + f'/sim_{job_number}', (theta, x))