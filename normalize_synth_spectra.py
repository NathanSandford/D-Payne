import numpy as np
import utils
import multiprocessing

# D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'

inputdir = D_PayneDir + '/spectra/synth_spectra/'
specfile = 'convolved_synthetic_spectra_MIST.npz'
OutputDir = inputdir

# Restore Spectra
print("Reading in synthetic spectra...")
temp = np.load(inputdir+specfile)
spectra = temp['spectra']
try:
    norm_spectra_true = temp['norm_spectra_true']
except KeyError:
    norm_spectra_true = temp['norm_spectra']
wavelength = temp['wavelength']
labels = temp['labels']
model = temp['model']
temp.close()

# Restore DEIMOS continuum pixels
print("Loading continuum region...")
cont_reg = utils.load_deimos_cont_pixels()

# Calculate matrix of distances between wavelengths
print("Calculating distance matrix...")
wavelength_diff_matrix = wavelength[:, np.newaxis] - wavelength


def normalize_spectra(i):
    cont_spectrum = \
        utils.get_deimos_continuum(spectra[i], spec_err=None,
                                   wavelength=wavelength,
                                   cont_pixels=cont_reg,
                                   wavelength_diff_matrix=wavelength_diff_matrix)
    norm_spec = spectra[i] / cont_spectrum
    print("Normalized spectrum number %d" % i)
    return(norm_spec)


pool = multiprocessing.Pool(multiprocessing.cpu_count())
norm_spectra_approx = pool.map(normalize_spectra, range(spectra.shape[0]))

# save the convolved spectra and their labels
print("Saving Normalized spectra to %s" % (specfile))
np.savez(OutputDir + specfile,
         spectra=spectra,
         norm_spectra_true=norm_spectra_true,
         norm_spectra_approx=norm_spectra_approx,
         wavelength=wavelength, labels=labels, model=model)

print("Normalization completed!")
