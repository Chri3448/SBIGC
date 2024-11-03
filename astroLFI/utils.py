#!/usr/bin/env python3
"""
File: utils.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Helper functions for astroLFI
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import yaml
import re
from astropy import cosmology
import scipy.interpolate as interp
import scipy.integrate as integrate
import astropy.units as u

def get_loader():
    """Create loader to process yaml files."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader

def setup_cosmology(H0=67.4, omegam0=0.32, omegade0=0.68):
    """Setup the assumed cosmology."""
    cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=omegam0)

    return cosmo

def power_law_approx_integrator(xs, ys, cumulative=False):
    """Compute the integral using the approximation that ys are power-law-behaved (i.e. compute the integral in log-log space.

    :xs: x-values 
    :ys: y-values 
    :cumulative: whether to return the cumulative integral or not [default: False]
    :returns: the value of the integral. If cumulative is True, returns an array of length len(xs)
    """
    ms = np.diff(np.log10(ys)) / np.diff(np.log10(xs))
    ns = (np.log10(ys[:-1]) - ms * np.log10(xs[:-1]))

    integral_split = np.where(np.isclose(ms, -1.0), 10**ns * np.log10(xs[1:] / xs[:-1]), 10**ns /     (ms + 1.0) * (xs[1:]**(ms + 1.0) - xs[:-1]**(ms + 1.0)))

    if cumulative is True:
        return np.concatenate(([0],np.cumsum(integral_split)))

    return integral_split.sum()

def differential_comoving_volume_interpolator(cosmo, zmin=0.01, zmax=10):
    """Construct an interpolator of the differential comoving volume.

    Provided to decrease computation time when integrating over the volume element.

    :param cosmo: astropy cosmology object
    :param zmin: minimum redshift for interpolator
    :param zmax: maximum redshift for interpolator

    :returns: interpolator of dV/dzdomega as a function of z  [Mpc**3 / sr]
    """
    zs = np.logspace(np.log10(zmin), np.log10(zmax), num=250)
    dVdzs = cosmo.differential_comoving_volume(zs).value

    return interp.interp1d(zs, dVdzs)

def solid_angle_of_sky_with_galactic_center_removed(galactic_plane_latitude_cut):
    return 2 * 2 * np.pi * (1 - np.cos(np.abs(np.pi/2 - galactic_plane_latitude_cut)))

def luminosity_distance(cosmo, redshifts):
    return cosmo.luminosity_distance(redshifts).to(u.cm).value 

def convert_observed_flux_and_luminosity(cosmo, *, flux=None, luminosity=None, alpha, redshift, Emin):
    """Convert between the observed flux and the rest frame luminosity.

    If both are provided, defaults to converting flux to luminosity

    :param cosmo: astropy cosmology
    :param flux: observed flux [photons cm^{-2} s^{-1}]
    :param luminosity: rest-frame luminosity [erg s^{-1}]
    :param alpha: photon spectral index
    :param redshift: redshift of source
    :param Emin: minimum energy

    :returns: If flux is provided, rest-frame luminosity is returned. If luminosity is provided, observed flux is returned.
    """
    conversion_factor = 4 * np.pi * luminosity_distance(cosmo, redshift)**2 * (alpha - 1) / (1 + redshift)**(2 - alpha) * Emin 
    if flux is not None:
        return conversion_factor * flux
    elif  luminosity is not None:
        return luminosity / conversion_factor
    else:
        print('Must provide either luminosity or flux.')

def recursive_dictionary_check(dic: dict, purpose='remove objects') -> dict:
    """Recurse through dictionaries for whichever purpose.

    Only implemented for removing functions from the dict so the dict can be serialized better.

    :param dic: dictionary to recurse through 
    :type dic: dict
    :param purpose: what to do with the dictionary

    :returns: dictionary modified in whichever way indicated by purpose
    :rtype: dict
    """
    if purpose == 'remove objects':
        for key, value in dic.items():
            if isinstance(value, dict):
                recursive_dictionary_check(value, purpose=purpose)
            elif callable(value):
                dic[key] = str(type(value))
                dic[key] = getattr(value, '__name__', repr(value))
        return dic
    else:
        raise NotImplementedError('Unrecognized purpose for dictionary recursion. Valid option is "remove objects".')


