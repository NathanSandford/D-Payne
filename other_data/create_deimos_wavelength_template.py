'''
Code to generate wavelenth template for DEIMOS.
It really should take into account DEIMOS's LSF.
'''

import numpy as np

lambda_min = 6250  # AA
lambda_max = 9500  # AA
wavelength_template = np.arange(lambda_min, lambda_max, 0.3)

np.savez('deimos_wavelength.npz', wavelength=wavelength_template)
