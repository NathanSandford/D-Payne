import numpy as np
import utils

#D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'

inputdir = D_PayneDir + '/spectra/synth_spectra/'
specfile = 'convolved_synthetic_spectra_MIST.npz'
#specfile = 'convolved_synthetic_spectra_kareem.npz'
RGB_index = 0

# Restore Spectra
temp = np.load(inputdir+specfile)
wavelength = temp['wavelength']
norm_spectra = temp['norm_spectra'][RGB_index]
labels = temp['labels'][RGB_index]
temp.close()

cont_reg = utils.get_continuum_pixels(wavelength,norm_spectra,0.96)

np.save('../other_data/deimos_cont_pixels.npy',cont_reg)
