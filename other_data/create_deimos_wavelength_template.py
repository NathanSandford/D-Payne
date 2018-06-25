'''
Code to generate wavelenth template for DEIMOS.
'''

import numpy as np

lambda_min = 6250  # AA
lambda_max = 9500  # AA
R_res = 6500

wavelength_template = []
wavelength_template.append(lambda_min)
wavelength_now = lambda_min
while wavelength_now < lambda_max:
    wavelength_now += wavelength_now/R_res
    if wavelength_now < lambda_max:
        wavelength_template.append(wavelength_now)
wavelength_template = np.array(wavelength_template)

wavelength_template =

np.savez('deimos_wavelength.npz', wavelength=wavelength_template)
