import numpy as np
import healpy as hp
import pickle as pk
import torch
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
import os
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

parameter_range = [[0., 0., 10000., 0.], [0.07, 2., 100000., 10.]]
prior = utils.BoxUniform(low = parameter_range[0], high = parameter_range[1])

load_posterior = False
load_sims = True
run = 'run7-1_nonpoisson_1m/'
numsims = '1000000sims'

posterior_path = 'posteriors/' + run
posterior_file = posterior_path + numsims
sim_dir = 'simulations/' + run
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
            if i == 10000:
                break
    else:
        simulator, prior = prepare_for_sbi(simulator, prior)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=10)
    inference = SNPE(prior=prior)
    #inference = SNPE(prior=prior, device='cuda')
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    os.makedirs(posterior_path, exist_ok=True)
    np.save(posterior_file, posterior, allow_pickle=True)