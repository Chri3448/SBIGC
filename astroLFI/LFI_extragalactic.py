import numpy as np
import sys
import os
import shutil
import pdb
import scipy
import astropy.units as u
import healpy as hp
import yaml
import copy
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils import process_prior
import dill
from pathlib import Path
from .LFI_unbinned import LFI_unbinned
from .utils import get_loader, setup_cosmology, differential_comoving_volume_interpolator, recursive_dictionary_check, solid_angle_of_sky_with_galactic_center_removed

from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")

class LFI_EG(LFI_unbinned):
    """LFI_EG."""
    def __init__(self, yaml_file, default_yaml=Path(__file__).parent / 'default_values.yaml'):
        """__init__.

        :param yaml_file: Configuration file
        :param default_yaml: Default configuration value that provides any settings not provided by the user
        """
        # set output width for output
        super().__init__(yaml_file)
        self.differential_comoving_volume_interp = differential_comoving_volume_interpolator(self.cosmo)

    #####################
    ##### INFERENCE #####
    #####################
    def run_sbi(self):
        self._print_line()
        print(f'Running SBI for {self.job_name}. Output will be stored in {self.output_path}')
        self._print_line()

        print(f'Checking simulator and prior...')
        simulator, prior = prepare_for_sbi(self.simulator, self.priors)
        print(f'...simulator and prior are good to go!')
        self._print_line()

        inference = SNPE(prior)
        
        print('Beginning simulations...')
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=self.number_of_sims, num_workers=1)
        print(f'Simulations complete! Sky simulated {self.number_of_sims} times')
        self._print_line()

        inference = inference.append_simulations(theta, x)

        return inference

    def run_sbi_restrict_prior(self, num_rounds, return_sims=False):
        from sbi.utils import RestrictionEstimator
        self._print_line()
        print(f'Running SBI and restricting the priors for {self.job_name}. Output will be stored in {self.output_path}')
        self._print_line()

        print(f'Checking simulator and prior...')
        simulator, prior = prepare_for_sbi(self.simulator, self.priors)
        print(f'...simulator and prior are good to go!')
        self._print_line()

        restriction_estimator = RestrictionEstimator(prior=prior)
        proposals = [prior]

        print('Beginning simulations...')
        for r in range(num_rounds):
            theta, x = simulate_for_sbi(simulator, proposals[-1], self.number_of_sims)
            restriction_estimator.append_simulations(theta, x)
            if (
                r < num_rounds - 1
            ):  # training not needed in last round because classifier will not be used anymore.
                classifier = restriction_estimator.train()
            proposals.append(restriction_estimator.restrict_prior())

        all_theta, all_x, _ = restriction_estimator.get_simulations()

        inference = SNPE(prior=prior).append_simulations(all_theta, all_x)

        print(f'Simulations complete! Sky simulated {len(all_theta)} times')
        self._print_line()

        if return_sims:
            return all_theta, all_x
        return inference

    #####################
    ##### SIMULATOR #####
    #####################
    def simulator(self, sample):
        self.print_output(f'Simulating extragalactic gamma ray sky', kind='verbose', header=2, footer=1)
        # parse sample for each source
        for source_name, source in self.sources.items():
            sample_slice = sample[source.sample_index_min:source.sample_index_max].numpy()
            sample_dict = dict(zip(source.sample_slice_names, sample_slice))
            sample_dict.update(source.fixed_params)
            source.sample = sample_dict
            
            self.print_output(f'Parameters getting passed to {source_name}: {source.sample}', kind='debug')

        # create sources
        source_info = self.create_sources()

        # generate photons
        photon_info = self.generate_photons_from_sources(source_info)

        # apply observational effects, mock_observe assumes energies in MeV so we need to convert
        photon_info['energies'] *= 1000
        photon_info = self.mock_observe(photon_info, self.obs_info)
        photon_info['energies'] /= 1000

        theta = np.pi/2 - self.galactic_plane_latitude_cut
        remove_photons_in_galactic_plane = (photon_info['angles'][:, 0] > theta) & (photon_info['angles'][:, 0] < np.pi - theta)
        photon_info['angles'] = photon_info['angles'][~remove_photons_in_galactic_plane]
        photon_info['energies'] = photon_info['energies'][~remove_photons_in_galactic_plane]

        # get summary statistic
        self.print_output(f'Computing summary statistic', kind='verbose', header=1)
        self.summary_properties.update({'Emin': self.obs_info['Emin'], 'Emax': self.obs_info['Emax']})
        summary_stat = self.get_summary(photon_info, summary_properties=self.summary_properties)
        self.print_output(f'Summary statistic computed: {summary_stat}', kind='verbose')

        return summary_stat
       
    def create_sources(self) -> dict:
        '''Create a list of sources for each source class, where each source has a radial distance, mass, and luminosity

        :returns: Source info for all sources
        :rtype: dict
        '''
        source_info = {}

        self.print_output(f'Creating all sources', kind='verbose', header=1, footer=1)

        # loop through sources
        for source_name, source in self.sources.items():
            # create dictionary for each source class considered
            source_info[source_name] = {}

            # do not need to generate individual sources for isotropic gamma rays
            if source.type == 'isotropic':
                continue

            self.print_output(f'Creating {source_name}', kind='verbose', header=1)

            source_info[source_name] = source.generate_sources()

            self.print_output(f'luminosities: {source_info[source_name]["luminosities"]}', kind='debug', prepend=True)
            self.print_output(f'distances: {source_info[source_name]["distances"]}', kind='debug', prepend=True)
            self.print_output(f'angles:, {source_info[source_name]["angles"]}', kind='debug', prepend=True)
            self.print_output(f'single_photon_angles: {source_info[source_name]["single_photon_angles"]}', kind='debug', prepend=True)
            self.print_output(f'single_photon_distances: {source_info[source_name]["single_photon_distances"]}', kind='debug', prepend=True)
            
        if self.verbose:
            self.print_output(f'Sources successfully created', kind='verbose')
            for source in source_info:
                self.print_output(f'Source: {source}', kind='verbose')
                self.print_output(f'Number of multi-photon sources created: {np.size(source_info[source_name]["luminosities"])}', kind='verbose', prepend=True)
                self.print_output(f'Number of single-photon sources created: {np.size(source_info[source_name]["single_photon_distances"])}', kind='verbose', prepend=True)

        return source_info
 
    def generate_photons_from_sources(self, source_info):
        '''Generate the photons from all of the sources.

        :param source_info: dictionary containing all of the information about the source distances, position/angles, and luminosity
        :type source_info: dict

        '''
        #TODO: figure out how to create an empty numpy array of arbitrary shape
        # create arrays to store info about all photons
        angles = np.array([[0.0, 0.0]])
        energies = np.array([])

        # create array for sampling the photon spectra
        energy_array = np.logspace(np.log10(self.obs_info['Emin']), np.log10(self.obs_info['Emax']), num=(self.obs_info['Emax'] - self.obs_info['Emin'] + 1)*5)

        self.print_output(f'Creating photons for all sources', kind='verbose', header=2)

        # loop through all sources and create photons
        for source_name, source_dict in source_info.items():
            if len(source_dict['luminosities']) == 0:
                continue
            self.print_output(f'Generating photons for {source_name}', kind='verbose', header=1)

            # only generate photons when there are multi-photon sources
            energy, angle = self.sources[source_name].generate_photons(source_dict, energy_array)

            self.print_output(f'number of photons {len(energy)}', kind='debug', prepend=True)
            self.print_output(f'angles {angle}', kind='debug', prepend=True)
            self.print_output(f'energies {energy}', kind='debug', prepend=True)
 
            energies = np.concatenate((energies, energy))
            angles = np.concatenate((angles, angle))

        # create dictionary of generated photons
        valid = True
        if len(energies) == 0:
            valid = False
            self.print_output('No photons created. Assuming invalid simulation.', kind='verbose')
        # ugly to get rid of garbage first entry of angles. see todo above
        photon_info = {'angles': angles[1:], 'energies': energies, 'valid': valid}

        self.print_output(f'Photons successfully created: {len(photon_info["energies"])}', kind='verbose', header=1, footer=1)

        return photon_info
    
if __name__ == '__main__':
    trial = LFI_EG(sys.argv[1]) 
    # trial.sources['blazars_all']['sample'] = dict(zip(['log10A', 'log10Lstar', 'gamma1', 'gamma2', 'p1', 'p2', 'zc', 'alpha', 'log10Lalpha', 'beaming_factor'], [-4, 44, 0.9, 2.4, 5, -1.1, 2.0, 0.4, 44, 2]))
    # source_info = trial.create_sources()
    # photon_info = trial.generate_photons_from_sources(source_info)
    # photon_info = trial.mock_observe(photon_info, trial.obs_info)
    trial.number_of_sims = 1000
    theta, x = trial.run_sbi_restrict_prior(10, return_sims=True)

    with open(trial.output_path + "simulations.pickle", 'wb') as f:
        pickle.dump((theta, x), f)

    exit()
    inference = trial.run_sbi_restrict_prior(10)

    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    with open(trial.output_path + "posterior.pickle", 'wb') as f:
        pickle.dump(posterior, f)
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(nrows=3, figsize=(12, 5))
    NSIDE = 64
    threshold1 = 5
    threshold2 = 20
    ebin1 = photon_info['energies'] < threshold1
    ebin2 = (photon_info['energies'] >= threshold1) & (photon_info['energies'] < threshold2)
    ebin3 = photon_info['energies'] >= threshold2

    # for title, ebin, ax in zip([f"E<{threshold1}", f"{threshold1}<E<{threshold2}", f"E>{threshold2}"], [ebin1, ebin2, ebin3], axs.flatten()):
    for title, ebin in zip([f"E<{threshold1}", f"{threshold1}<E<{threshold2}", f"E>{threshold2}"], [ebin1, ebin2, ebin3]):
        pixels = hp.ang2pix(NSIDE, photon_info['angles'][:, 0][ebin], photon_info['angles'][:, 1][ebin])
        hist, hist_edges = np.histogram(pixels, np.arange(0, hp.nside2npix(NSIDE) + 1)-0.5)

        hp.mollview(hist, title=title)
        plt.savefig(f'./trial_plot_{title}.jpg')
        plt.show()
        
