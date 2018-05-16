'''
Adapted from Kareem's:
binspec/train_NN/convolve_highrez_spectra_to_given_R.py.

This code reads in a batch of very high resolution spectra and degrades them
to a lower resolution, assuming a Gaussian line-spread function. The main use
case is that we produce synthetic spectra (from an updated version of the
Kurucz line list by default) at R~300,000 and need to convolve them the
resolution of DEIMOS before training the NN spectral model. Note, the high-res
spectra are normalized.

Since we are doing ab-initio fitting, we should be using the correct
(non-Gaussian) LSF from DEIMOS, but I have not got around to it yet.
'''

# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy import interpolate
import multiprocessing
import utils
wavelength_template = utils.load_wavelength_array()

# D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne/'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'
SynthSpectraDir = '/global/home/users/nathan_sandford/kurucz/synthetic_spectra/'
FileName = 'synthetic_spectra_MIST.npz'
# FileName = 'synthetic_spectra_kareem.npz'
OutputDir = D_PayneDir + 'spectra/synth_spectra/'

# restore spectra
print("Reading in synthetic spectra...")
temp = np.load(SynthSpectraDir + FileName)
# This is the R~300,000 wavelength, *not* our default grid.
wavelength = temp['wavelength']
spectra = temp['spectra']
continuum = temp['continuum']
norm_spectra = spectra / continuum
labels = temp['labels']
model = temp['model']
temp.close()


def sparsify_matrix(lsf):
    '''
    This just speeds up the computation,
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


R_res = 6500  # for DEIMOS
start_wavelength = wavelength_template[0]  # 6250
end_wavelength = wavelength_template[-1]  # 9500

# interpolation parameters. The interpolation resolution just needs to be
# significantly better than the final resolution of the wavelength grid
inv_wl_res = 100
wl_res = 1./inv_wl_res
wl_range = end_wavelength - start_wavelength

# make convolution grid
print("Making convolution grid...")
wavelength_run = wl_res*np.arange(wl_range/wl_res + 1) + start_wavelength

# determines where we can cut off the convolution kernel.
# multiply by R_apogee / R_deimos to keep template_width large enough
template_width = np.median(np.diff(wavelength_template)) * 22500 / R_res

# how many kernel bin to keep
R_range = int(template_width/wl_res + 0.5)*5

# pad wavelength with zeros for convolution
wavelength_tmp = np.concatenate([np.zeros(R_range), wavelength_run,
                                np.zeros(R_range)])

# create convolution matrix. Each column is a Gaussian kernel with
# different FWHM, with the FWMH equal to  FWHM = lambda/R_res
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
    convolve a single spectrum. Pass this to multiprocessing.
    '''
    # interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, spectra[c1, :])
    full_spec = f_flux_spec(wavelength_run)

    # convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    # return convolved spectrum
    print('convolved spectrum number %d' % c1)
    return f_flux_1D(wavelength_template)


def convolve_norm_spectrum(c1):
    '''
    convolve a single spectrum. Pass this to multiprocessing.
    '''
    # interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, norm_spectra[c1, :])
    full_spec = f_flux_spec(wavelength_run)

    # convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    # return convolved spectrum
    print('convolved spectrum number %d' % c1)
    return f_flux_1D(wavelength_template)


# convolve multiple spectra in parallel
print("Convolving spectra...")
pool = multiprocessing.Pool(multiprocessing.cpu_count())
spectra = pool.map(convolve_spectrum, range(spectra.shape[0]))
norm_spectra = pool.map(convolve_norm_spectrum, range(norm_spectra.shape[0]))

# save the convolved spectra and their labels
print("Saving Convolved spectra to %s" % ('convolved_'+FileName))
np.savez(OutputDir + 'convolved_'+FileName,
         spectra=spectra, norm_spectra=norm_spectra,
         wavelength=wavelength_template, labels=labels, model=model)

print("Convolution completed!")
