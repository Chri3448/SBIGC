# SBIGC
Code developed for the paper "Applying Simulation-Based Inference to Spectral and Spatial Information from the Galactic Center Gamma-Ray Excess" [2402.04549]

The LFI_galactic_center.py file in the astroLFI directory contains the class, which would later become AEGIS, used for generating simulations of the gamma-ray map of the galactic center due to potential millisecond pulsars, diffuse dark matter annihilation, and known astrophysical sources. These sources are contained in the sources directory. These simulations were used to train a neural network model using the sbi package from Mackelab. The files used for training can be found in the gc_jobs directory.
