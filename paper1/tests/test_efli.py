import numpy as np
import EPDFABC as epdfabc
from scipy.stats import poisson
import elfi
import pdb

def pdf_source1(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    #Poisson PDF with mean specified by params[0]
    return poisson.pmf(counts, params[0])

def pdf_source2(params, counts, pixel_index = None):
    #define a PDF here as a function of parameters
    return poisson.pmf(counts, params[1]+10.)    

def spectrum_source1(params, energy):
    #define a spectrum here as a function of parameters
    #doesn't need to be normalized
    return energy**params[2]

def spectrum_source2(params, energy):
    #define a spectrum here as a function of parameters
    return energy**(params[3]+0.1)

def lnprior(params):
    return 0.


PDF_and_spectra = [[pdf_source1, spectrum_source1], [pdf_source2, spectrum_source2]]
isotropic = True
energy_range = np.array([1., 100.])
count_range = np.array([0., 150.])
param_names = ['barc1', 'barc2', 'a1', 'a2']
param_labels = ['\\bar{C}_1', '\\bar{C}_2', '\\alpha_1', '\\alpha_2']
param_min = np.array([0., 0., -3., -3.])
param_max = np.array([100., 100., 0., 0.])
param_range = [param_min, param_max]

param_true = 0.5*(param_min + param_max)

counts = np.arange(0,count_range[1])
energies = np.linspace(energy_range[0], energy_range[1], num = 100)

# initialize the EPDFABC object
my_abc = epdfabc.EPDFABC(param_range)

#Setup for binned analysis
my_abc.setup_binned(PDF_and_spectra, isotropic, count_range, energy_range, verbose = True)
my_abc.add_lnprior(lnprior)

#Create mock data with "true" parameters
N_E_bins = 5
N_pix = 20

priors = []
for pi in range(0,len(param_min)):
    priors.append(elfi.Prior('uniform', param_min[pi], param_max[pi]-param_min[pi], name = param_names[pi]))
    
y0 = my_abc.generate_mock_data_binned_ELFI(param_true[0], param_true[1], param_true[2], param_true[3], N_pix = N_pix, N_energy = N_E_bins, epdf_object = my_abc)

from functools import partial

fn_simulator = partial(my_abc.generate_mock_data_binned_ELFI, N_pix = N_pix, N_energy = N_E_bins, epdf_object=my_abc)
sim = elfi.Simulator(fn_simulator, *priors, observed=y0)

fn_summary = partial(my_abc.get_summary_from_binned)
S1 = elfi.Summary(fn_summary, sim)

d = elfi.Distance('euclidean', S1)

rej = elfi.Rejection(d, batch_size=2)
res = rej.sample(10, threshold=50)

pdb.set_trace()
