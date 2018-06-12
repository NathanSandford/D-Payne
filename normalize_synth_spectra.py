import numpy as np
import utils
import multiprocessing

'''
Print updates?
'''
verbose = False


'''
Set Directories
'''
D_PayneDir = utils.D_PayneDir
inputdir = D_PayneDir + '/spectra/synth_spectra/'
specfile = 'convolved_synthetic_spectra_MIST.npz'
OutputDir = inputdir


'''
Restore Spectra
'''
if verbose:
    print("Reading in synthetic spectra...")
temp = np.load(inputdir+specfile)
# Unnormalized Spectra
spectra = temp['spectra']
# Normalized Spectra (from theoretical continuum)
try:
    norm_spectra_true = temp['norm_spectra_true']
except KeyError:
    norm_spectra_true = temp['norm_spectra']
# Wavelength (should already be on DEIMOS grid)
wavelength = temp['wavelength']
# Labels: [alpha/Fe], [Mg/H], ..., [Fe/H], Teff, logg
labels = temp['labels']
# Model name from Kurucz/Atlas12 models
model = temp['model']
temp.close()


'''
Restore DEIMOS continuum pixels
'''
if verbose:
    print("Loading continuum region...")
cont_reg = utils.load_deimos_cont_pixels()


'''
Calculate matrix of distances between wavelengths
'''
if verbose:
    print("Calculating distance matrix...")
wavelength_diff_matrix = wavelength[:, np.newaxis] - wavelength


def normalize_spectra(i):
    cont_spectrum = \
      utils.get_deimos_continuum(spectra[i], spec_err=None,
                                 wavelength=wavelength,
                                 cont_pixels=cont_reg,
                                 wavelength_diff_matrix=wavelength_diff_matrix)
    norm_spec = spectra[i] / cont_spectrum
    if verbose:
        print("Normalized spectrum number %d" % i)
    return(norm_spec)


'''
Normalize in Parallel
'''
pool = multiprocessing.Pool(multiprocessing.cpu_count())
norm_spectra_approx = pool.map(normalize_spectra, range(spectra.shape[0]))


'''
Save the convolved spectra and their labels
'''
if verbose:
    print("Saving Normalized spectra to %s" % (specfile))
np.savez(OutputDir + specfile,
         spectra=spectra,
         norm_spectra_true=norm_spectra_true,
         norm_spectra_approx=norm_spectra_approx,
         wavelength=wavelength, labels=labels, model=model)

if verbose:
    print("Normalization completed!")
