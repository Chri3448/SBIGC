#!/usr/bin/env python3
"""
File: extragalactic_point_sources.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: 
"""
import numpy as np
from ..utils import convert_observed_flux_and_luminosity, solid_angle_of_sky_with_galactic_center_removed

class EG_PS():
    def __init__(self, LFI_obj, dict_from_yaml):
        self.LFI_obj = LFI_obj
        self.sample_2D_PDF = LFI_obj.draw_from_2D_pdf

        for key, item in dict_from_yaml.items():
            setattr(self, key, item)

        if self.number_density == 'LDDE':
            self.number_density_function = self.LDDE
        else:
            raise NotImplementedError(f"Unrecognized LF evolution model {self.redshift_evolution}. Valid options are: LDDE.")

    def get_param_names(self):
        return self.number_density_function([], get_helper=True)

    def generate_sources(self):
        luminosities, distances, single_photon_distances = self.draw_luminosities_and_radii()

        num_sources = np.size(luminosities)
        num_single_photon_sources = np.size(single_photon_distances)

        # generate angles 
        angles = self.draw_angles(num_sources)

        source_dict = {}
        source_dict['luminosities'] = luminosities
        source_dict['distances'] = distances
        source_dict['angles'] = angles
        
        # generate angles for single photon sources and identify sources outside the galactic plane
        if num_single_photon_sources > 0:
            single_photon_angles = self.draw_angles(num_single_photon_sources)

            # single photon source information
            source_dict['single_photon_distances'] = single_photon_distances
            source_dict['single_photon_angles'] = single_photon_angles
        else:
            # if there are no single photon sources, set the parameters to None
            source_dict['single_photon_distances'] = None
            source_dict['single_photon_angles'] = None

        return source_dict

    def generate_photons(self, source_dict, energy_array):
        # calculate the expected number of photons from each source
        alpha = self.alpha_photon_spectrum
        mean_photon_counts = self.LFI_obj.exposure * convert_observed_flux_and_luminosity(self.LFI_obj.cosmo, luminosity=source_dict['luminosities'], alpha=alpha, redshift=source_dict['distances'], Emin=self.LFI_obj.obs_info['Emin'])
        photon_counts = np.random.poisson(mean_photon_counts)

        if photon_counts.sum() > self.LFI_obj.maximum_number_of_photons:
            self.LFI_obj.print_output(f'Too many photons expected ({photon_counts.sum() + len(energies)}). Invalid simulation.', kind='verbose')
            return {'angles': np.array([[0.0, 0.0]]), 'energies': np.array([0.0]), 'valid': False}

        self.LFI_obj.print_output(f'Total photons drawn from multi-photon sources: {photon_counts.sum()}', kind='verbose', prepend=True)
        self.LFI_obj.print_output(f'Mean photon counts from multi-photon sources: {mean_photon_counts}', kind='debug', prepend=True)
        self.LFI_obj.print_output(f'Drawn photons: {photon_counts}', kind='debug', prepend=True)

        # create an array of all photons from all sources
        photon_angles = np.repeat(source_dict['angles'],  photon_counts, axis=0)

        #TODO: allow for different alpha values
        #TODO: generalize photon spectrum
        dndE = energy_array**-alpha
        # dndE /= self.power_law_approx_integrator(energy_array, dndE)
        dndE /= dndE.sum()

        #TODO: could probably be pulling directly from the pdf
        #TODO: first argument of draw_from_pdf is unused
        energy_indices = self.LFI_obj.draw_from_pdf(None, dndE, photon_counts.sum())

        # add photon info to arrays
        angles = photon_angles
        energies = energy_array[energy_indices]

        # add in single photon source angles. inserted after multi-photon sources
        if source_dict['single_photon_angles'] is not None:
            # if there are single photon sources, add their info as well
            angles = np.concatenate((angles, source_dict['single_photon_angles']))
            single_photon_energy_indices = self.LFI_obj.draw_from_pdf(np.arange(0, len(energy_array)), dndE, len(source_dict['single_photon_distances']))
            energies = np.concatenate((energies, energy_array[single_photon_energy_indices]))

        self.LFI_obj.print_output(f'there are {(energies > energy_array[0]).sum()} photons with energy larger than the minimum energy', kind='debug', prepend=True)

        return energies, angles

    def draw_luminosities_and_radii(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Choose luminosities and radii for the provided source.

        :param params: dictionary containing the astrophysical parameters about the source (including number density, L_min, L_max, L_bins, distance_min, distance_max, distance_bins
        :type source: dict

        :returns: 3-tuple containing lumionsities of each source, distances of each source, distances of single photon sources
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        # create luminosity and distance arrays to sample
        luminosities = np.logspace(np.log10(self.L_min), np.log10(self.L_max), num=self.L_bins)
        distances = np.logspace(np.log10(self.distance_min), np.log10(self.distance_max), num=self.distance_bins)

        # sample the number density on an (L, distance) grid
        number_density = self.number_density_function(self.sample)(luminosities[:-1][:, np.newaxis], distances[:-1][np.newaxis, :])

        # get number of sources in each (L, d) bin by integrating number density
        dlog10L = np.diff(np.log10(luminosities))
        dD = np.diff(distances)

        # compute the integrand on the sampling grid
        integrand = number_density * solid_angle_of_sky_with_galactic_center_removed(self.LFI_obj.galactic_plane_latitude_cut) * self.LFI_obj.differential_comoving_volume_interp(distances[:-1])[np.newaxis, :] * dlog10L[:, np.newaxis] * dD[np.newaxis, :]

        C = self.LFI_obj.exposure * convert_observed_flux_and_luminosity(self.LFI_obj.cosmo, luminosity=luminosities[:-1][:, np.newaxis], alpha=self.alpha_photon_spectrum, redshift=distances[:-1][np.newaxis, :], Emin=self.LFI_obj.obs_info['Emin'])

        self.LFI_obj.print_output(f'Binomial trick being used at {np.sum(C < self.LFI_obj.binomial_trick_epsilon)}/{C.flatten().size} points in sampling space', kind='debug', prepend=True)
        self.LFI_obj.print_output(f'maximum mean photon value {C.max()}', kind='debug', prepend=True)
        self.LFI_obj.print_output(f'C: {C}', kind='debug', prepend=True)

        # set all bins that don't match binomial trick criterion to zero
        C[C > self.LFI_obj.binomial_trick_epsilon] = 0

        # calculate binomial trick term
        p = C * np.exp(-C)

        # get single photon distances
        num_single_photon_sources_at_distance = np.round(np.sum(integrand*p, axis=0)).astype(int)
        single_photon_distances = np.repeat(distances[:-1], num_single_photon_sources_at_distance)

        # don't sample the binomial trick region again
        integrand[C != 0] = 0

        self.LFI_obj.print_output(f'Expected number of sources: {integrand.sum()}', kind='debug', prepend=True)
        if integrand.sum() >= self.LFI_obj.maximum_number_of_sources:
            self.LFI_obj.print_output(f'Number of expected sources is too large ({integrand.sum()}). Assuming parameter sample is non-physical and creating no sources', kind='debug', prepend=True)
            return [], [], []

        if np.any(integrand != 0):
            luminosity_indices, distance_indices = self.sample_2D_PDF(integrand, Ndraws=0)
        else:
            self.LFI_obj.print_output('Danger: All sources are considered single photon for input parameters...something may have gone wrong', kind='verbose')
            luminosity_indices, distance_indices = [], []

        return luminosities[luminosity_indices], distances[distance_indices], single_photon_distances

    def draw_angles(self, num: int) -> np.ndarray:
        """Generate angles for isotropic sources on the sky.

        Note that the angular coordinates are in (colatitude, longitude) and so run from (0, pi) and (0, 2 pi).

        :param num: number of angular coordinates to draw
        :type num: int

        :return: numpy array of shape (num, 2) with colatitudes (thetas) in the first column and longitudes (phis) in the second [radians]
        :rtype: np.ndarray
        """
        angles = np.ones([num, 2])
        rands = np.random.rand(num) - 0.5
        theta = np.pi / 2 - self.LFI_obj.galactic_plane_latitude_cut
        rands[rands < 0] = - (np.abs(rands[rands < 0]) * 2 * (1 - np.cos(theta)) + np.cos(theta))
        rands[rands >= 0] = rands[rands >= 0] * 2 * (1 - np.cos(theta)) + np.cos(theta)
        angles[:,0] = np.arccos(rands)
        angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num)

        return angles


    # Number density functions take in sampled parameters in a list and return a function of phi(L, z)
    # They also must provide a helper dictionary flagged by get_helper that provides the ordering of the named parameters
    def double_power_law(self, L, *, A, Lstar, gamma1, gamma2, **kwargs):
        """Standard double power law function parametrizing AGN LF.

        :param L: Luminosity in [erg s^-1] to evaluate the LF
        :type L: float or array
        :param A: Amplitude of LF [Mpc^-3]
        :type A: float or array
        :param Lstar: luminosity break (i.e. knee) [erg s^-1]
        :type Lstar: float or array
        :param gamma1: faint end slope
        :type gamma1: float or array
        :param gamma2: bright end slope
        :type gamma2: float or array
        
        :returns: The value of the LF d\phi(L)/dlogL
        :rtype: float or array broadcast over input shapes
        """
        return A / ((L / Lstar)**gamma1 + (L / Lstar)**gamma2)

    def LADE(self, lade_params, **kwargs):
        """LADE evolutionary model of AGN LF.

        :param lade_params: Dictionary specifying the necessary components for LADE (A, Lstar, gamma1, gamma2, p1, p2, zc, d)
        :type lade_params: dict

        :returns: Double power law function with LADE evolution
        :rtype: function
        """
        A = lade_params['A']
        logLstar = lade_params['logLstar']
        gamma1 = lade_params['gamma1']
        gamma2 = lade_params['gamma2']
        p1 = lade_params['p1']
        p2 = lade_params['p2']
        zc = lade_params['zc']
        d = lade_params['d']

        #TODO: get_helper()

        k = (1 + zc)**p1 + (1 + zc)**p2

        def eta1(z):
            return 1 / k * (((1 + zc) / (1 + z))**p1 + ((1 + zc) / (1 + z))**p2)

        def etad(z):
            return 10**(d * (1 + z))

        return lambda L, z: self.double_power_law(L, A=A*etad(z), Lstar=(10**logLstar)/eta1(z), gamma1=gamma1, gamma2=gamma2)

    def LDDE(self, ldde_params, get_helper=False, **kwargs):
        """LDDE evolutionary model of AGN LF.

        :param ldde_params: Dictionary specifying the necessary components for LDDE (A, Lstar, gamma1, gamma2, p1, p2, zc, alpha, Lalpha)
        :type ldde_params: dict

        :returns: Double power law function with LDDE evolution, phi(L, z)
        :rtype: function
        """
        # provide a helper dictionary to parse priors
        if get_helper:
            LDDE_ordering = {param_name: num for num, param_name in enumerate(['log10A', 'log10Lstar', 'gamma1', 'gamma2', 'p1', 'p2', 'zc', 'alpha', 'log10Lalpha', 'beaming_factor'])}
            LDDE_ordering = ['log10A', 'log10Lstar', 'gamma1', 'gamma2', 'p1', 'p2', 'zc', 'alpha', 'log10Lalpha', 'beaming_factor']
            return LDDE_ordering

        # unpack sample
        # log10A, log10Lstar, gamma1, gamma2, p1, p2, zc, alpha, log10Lalpha, beaming_factor = ldde_params
        log10A = ldde_params['log10A']
        log10Lstar = ldde_params['log10Lstar']
        gamma1 = ldde_params['gamma1']
        gamma2 = ldde_params['gamma2']
        p1 = ldde_params['p1']
        p2 = ldde_params['p2']
        zc = ldde_params['zc']
        alpha = ldde_params['alpha']
        log10Lalpha = ldde_params['log10Lalpha']
        beaming_factor = ldde_params['beaming_factor']

        A = pow(10.0, log10A) / 10**beaming_factor
        Lstar = pow(10.0, log10Lstar) * 10**beaming_factor
        Lalpha = pow(10.0, log10Lalpha) * 10**beaming_factor

        # define evolution functions
        def z0(L):
            return np.where(L >= Lalpha, zc, zc * (L / Lalpha)**alpha)

        def ldde(L, z):
            return np.where(z <= z0(L), (1 + z)**p1, (1 + z0(L))**p1 * ((1 + z) / (1 +z0(L)))**p2)

        # return number density function
        return lambda L, z: self.double_power_law(L, A=A*ldde(L, z), Lstar=Lstar, gamma1=gamma1, gamma2=gamma2)


