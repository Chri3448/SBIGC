'''
Runs test on ABC analysis pipeline
'''

import numpy as np
import EPDFABC as epdfabc
from scipy.stats import poisson
import pdb

def pdf_source1(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    #Poisson PDF with mean specified by params[0]
    return poisson.pmf(counts, params[0])

def pdf_source2(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    return poisson.pmf(counts, params[1])    

def spectrum_source1(params, energy):
    #define a spectrum here as a function of parameters
    #doesn't need to be normalized
    return energy**params[2]

def spectrum_source2(params, energy):
    #define a spectrum here as a function of parameters
    return energy**params[2]    

def lnprior(params):
    return 0.

PDF_and_spectra = [[pdf_source1, spectrum_source1], [pdf_source2, spectrum_source2]]
isotropic = True
energy_range = np.array([1., 100.])
param_min = np.array([0., 0., -3., -3.])
param_max = np.array([100., 100., 0., 0.])
param_range = [param_min, param_max]

# initialize the EPDFABC object
my_abc = epdfabc.EPDFABC(param_range)

#Setup for binned analysis
my_abc.setup_binned(PDF_and_spectra, isotropic, energy_range, verbose = True)
my_abc.add_lnprior(lnprior)

#Create mock data with "true" parameters
param_true = 0.5*(param_min + param_max)
N_E_bins = 5
N_pix = 10
max_counts = 500
obs_data = my_abc.generate_mock_data_binned(param_true, N_pix, N_E_bins, max_counts, verbose = True)

#Get posterior using ABC
posterior_samples = my_abc.run_abc_binned(obs_data)

print("posterior samples = ", posterior_samples)
print(posterior_samples.shape)

print("Test completed!")
