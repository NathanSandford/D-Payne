'''
Code to determine continuum region using a theoretically normalized spectrum.

Continuum regions are determined such that the theoretically normalized
spectrum within 0.5 AA of that pixel does not deviate from 1 by more than 4%.
'''

import numpy as np
import utils

'''
Set Directories
'''
D_PayneDir = utils.D_PayneDir
inputdir = D_PayneDir + '/spectra/synth_spectra/'
specfile = 'convolved_synthetic_spectra_MIST.npz'

'''
Index of Theoretically Normalized Spectrum
'''
RGB_index = 0

'''
Restore Theoretically Normalized Spectra
'''
temp = np.load(inputdir+specfile)
wavelength = temp['wavelength']
norm_spectra_true = temp['norm_spectra_true'][RGB_index]
labels = temp['labels'][RGB_index]
temp.close()

cont_reg = utils.get_continuum_pixels(wavelength, norm_spectra_true, 0.96)

np.save('../other_data/deimos_cont_pixels.npy', cont_reg)
