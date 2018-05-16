'''
Code determins what regions of the spectrum to mask by comparing
the spectrum generated from the NN using the "True" labels of a star
with the observed spectrum of that star.

This should take into account both theoretical errors in the line lists
as well as observational/instrumental errors.

Currently this is using a high-resolution spectrum of the sun convolved down
to DEIMOS resolution. This is not ideal as it was not taken with DEIMOS
'''

import numpy as np
import utils
from model_spectra import get_spectrum_from_neural_net

# D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'

wavelength = utils.load_wavelength_array()

# Restore convolved spectra of the Sun and Arcturus
temp = np.load(D_PayneDir + '/spectra/convolved_masking_spectra.npz')
spectra_sol = temp['spectra_sol']
spectra_sol2 = temp['spectra_sol2']
spectra_arc = temp['spectra_arc']
temp.close()

# Restore NN
NN_coeffs = utils.read_in_neural_network(name='norm_spectra_approx')

'''
Solar Labels =
[alpha/Fe],[Mg/Fe],[Si/Fe],[S/Fe],[Ar/Fe],[Ca/Fe],[Ti/Fe],[Fe/H],Teff,logg
'''
labels_solar = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5780, 4.438])

# Generate Solar Spectrum from the Payne
spectra_sol_payne = get_spectrum_from_neural_net(labels_solar, NN_coeffs)

# Mask pixels with error > 2%
model_err = spectra_sol_payne - spectra_sol
mask = np.argwhere(np.abs(model_err) > 0.02)

np.save(D_PayneDir + 'other_data/payne_mask.npy', mask)
