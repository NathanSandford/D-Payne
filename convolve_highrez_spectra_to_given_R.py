'''
This code reads in a batch of very high resolution spectra and degrades them
to a lower resolution, assuming a Gaussian line-spread function. The main use
case is that we produce synthetic spectra (from an updated version of the
Kurucz line list by default) at R~300,000 and need to convolve them the
resolution of DEIMOS before training the NN spectral model.

Since we are doing ab-initio fitting, we should be using the correct
(non-Gaussian) LSF from DEIMOS, but I have not got around to it yet.

Adapted from Kareem's:
binspec/train_NN/convolve_highrez_spectra_to_given_R.py.
Much of the documentation is his.
'''

# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy import interpolate
import multiprocessing
import utils
# Restore DEIMOS Wavelength Grid
wavelength_template = utils.load_wavelength_array()


'''
Print updates?
'''
verbose = False


'''
Set Directories
'''
D_PayneDir = utils.D_PayneDir
KuruczSpectraDir = utils.KuruczSpectraDir
FileName = 'synthetic_spectra_MIST.npz'
OutputDir = D_PayneDir + 'spectra/synth_spectra/'


'''
Restore R~300,000 Spectra
'''
if verbose:
    print("Reading in synthetic spectra...")
temp = np.load(KuruczSpectraDir + FileName)
# This is the R~300,000 wavelength, NOT our default grid.
wavelength = temp['wavelength']
# Unnormalized Spectra
spectra = temp['spectra']
# Theoretical Continuum
continuum = temp['continuum']
# Theoretically Normalized Spectrum
norm_spectra_true = spectra / continuum
# Labels: [alpha/Fe], [Mg/H], ..., [Fe/H], Teff, logg
labels = temp['labels']
# Model name from Kurucz/Atlas12 models
model = temp['model']
temp.close()


def sparsify_matrix(lsf):
    '''
    This speeds up the computation,
    since the convolution matrix gets very large.
    '''
    nx = lsf.shape[1]
    diagonals = []
    offsets = []
    for ii in range(nx):
        offset = nx//2 - ii
        offsets.append(offset)
        if offset < 0:
            diagonals.append(lsf[:offset, ii])
        else:
            diagonals.append(lsf[offset:, ii])
    return sparse.diags(diagonals, offsets)


'''
DEIMOS Spectral Resolution & Coverage
'''
R_res = 6500
start_wavelength = wavelength_template[0]  # 6250
end_wavelength = wavelength_template[-1]  # 9500


'''
Interpolation Parameters:
The interpolation resolution just needs to be significantly better than the
final resolution of the wavelength grid
'''
inv_wl_res = 100
wl_res = 1./inv_wl_res
wl_range = end_wavelength - start_wavelength


'''
Make Convolution Grid
'''
if verbose:
    print("Making convolution grid...")
wavelength_run = wl_res*np.arange(wl_range/wl_res + 1) + start_wavelength


'''
Convolution Kernel Cutoff:
Multiply by R_apogee / R_deimos to keep template_width large enough
'''
template_width = np.median(np.diff(wavelength_template)) * 22500 / R_res


'''
Number of Kernel Bins to Keep
'''
R_range = int(template_width/wl_res + 0.5)*5


'''
Pad Wavelength w/ Zeros
'''
wavelength_tmp = np.concatenate([np.zeros(R_range), wavelength_run,
                                np.zeros(R_range)])


'''
Create Convolution Matrix:
Each column is a Gaussian kernel with FWMH equal to FWHM = lambda/R_res
'''
if verbose:
    print("Creating convolution matrix...")
conv_matrix = np.zeros((len(wavelength_run), 2*R_range+1))
for i in range(len(wavelength_run)):
    this_wl = wavelength_tmp[i:(i + 2*R_range + 1)] \
        - wavelength_tmp[i + R_range]
    # convert from FWHM to sigma.
    this_scale = wavelength_tmp[i+R_range]/(R_res*2.355)
    this_kernel = norm.pdf(this_wl, scale=this_scale)*wl_res
    conv_matrix[i, :] = this_kernel
conv_sparse = sparsify_matrix(conv_matrix)


def convolve_spectrum(c1):
    '''
    Convolve a single spectrum.
    Pass this to multiprocessing.
    '''
    # Interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, spectra[c1, :])
    full_spec = f_flux_spec(wavelength_run)

    # Convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    # Return convolved spectrum
    if verbose:
        print('convolved spectrum number %d' % c1)
    return f_flux_1D(wavelength_template)


def convolve_norm_spectrum(c1):
    '''
    Convolve a single theoretically normalized spectrum.
    Pass this to multiprocessing.
    '''
    # Interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, norm_spectra_true[c1, :])
    full_spec = f_flux_spec(wavelength_run)

    # Convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    # Return convolved spectrum
    if verbose:
        print('convolved spectrum number %d' % c1)
    return f_flux_1D(wavelength_template)


'''
Convolve in Parallel
'''
if verbose:
    print("Convolving spectra...")
pool = multiprocessing.Pool(multiprocessing.cpu_count())
spectra = pool.map(convolve_spectrum, range(spectra.shape[0]))
norm_spectra_true = pool.map(convolve_norm_spectrum,
                             range(norm_spectra_true.shape[0]))

'''
Save the convolved spectra and their labels
'''
if verbose:
    print("Saving Convolved spectra to %s" % ('convolved_'+FileName))
np.savez(OutputDir + 'convolved_'+FileName,
         spectra=spectra, norm_spectra_true=norm_spectra_true,
         wavelength=wavelength_template, labels=labels, model=model)

if verbose:
    print("Convolution completed!")
