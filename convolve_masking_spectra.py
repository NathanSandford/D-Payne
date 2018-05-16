'''
(Adapted from Kareem's binspec/train_NN/convolve_highrez_spectra_to_given_R.py.)
This code reads in a batch of very high resolution spectra and degrades them
to a lower resolution, assuming a Gaussian line-spread function. The main use
case is that we produce synthetic spectra (from an updated version of the Kurucz 
line list by default) at R~300,000 and need to convolve them the resolution of 
DEIMOS before training the NN spectral model. Note, the high-res spectra are not
normalized.
If we were doing ab-initio fitting, it would be important to use the correct
(non-Gaussian) LSF from DEIMOS, but since we just use the synthetic spectral
model to predict the continuum, this isn't important. 
For a few hundred ab-initio spectra, this runs on my laptop in a minute or two. 
I have not included the high-res model spectra that this operates on, as they're 
big files, but they're available up request if an example is needed. 
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from astropy.io import fits
from scipy import sparse
from scipy.stats import norm
from scipy import interpolate
import multiprocessing
import utils
wavelength_template = utils.load_wavelength_array()

#D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne/'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'
SpectraDir = D_PayneDir + '/spectra/'
SolarFileName = 'sao2010.solref'
ArcturusFileName = ' ardata.fits'
OutputDir = SpectraDir

# restore spectra
print("Reading in Solar spectrum...")
temp = np.genfromtxt(SpectraDir+'sao2010.solref')
wavelength_sol = temp.T[0] * 10
spectra_sol = temp.T[2]
print("Reading in Arcturus spectrum...")
temp = fits.open(SpectraDir+'ardata.fits')
wavelength_arc = temp[1].data['WAVELENGTH']
spectra_arc = temp[1].data['ARCTURUS']
spectra_sol2 = temp[1].data['SOLARFLUX']


def sparsify_matrix(lsf):
    '''
    This just speeds up the computation, since the convolution matrix gets very large. 
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

R_res = 6500 # for DEIMOS
start_wavelength = wavelength_template[0] #6250
end_wavelength = wavelength_template[-1] #9500

# interpolation parameters. The interpolation resolution just needs to be 
# significantly better than the final resolution of the wavelength grid
inv_wl_res = 100
wl_res = 1./inv_wl_res
wl_range = end_wavelength - start_wavelength

# make convolution grid
print("Making convolution grid...")
wavelength_run = wl_res*np.arange(wl_range/wl_res + 1)+ start_wavelength

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
    this_wl = wavelength_tmp[i:(i + 2*R_range + 1)] - wavelength_tmp[i + R_range]
    this_scale = wavelength_tmp[i+R_range]/(R_res*2.355) # convert from FWHM to sigma. 
    this_kernel = norm.pdf(this_wl, scale = this_scale)*wl_res
    conv_matrix[i, :] = this_kernel
conv_sparse = sparsify_matrix(conv_matrix)

def convolve_spectrum(wavelength,spectra):
    '''
    convolve a single spectrum. Pass this to multiprocessing. 
    '''
    # interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, spectra, bounds_error=False, fill_value=0)
    full_spec = f_flux_spec(wavelength_run)
    
    # convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    return f_flux_1D(wavelength_template)

print("Convolving Solar spectrum from %s..." %SolarFileName)
spectra_sol = convolve_spectrum(wavelength_sol,spectra_sol)
print("Normalizing Solar spectrum from %s..." %SolarFileName)
spectra_sol = spectra_sol / utils.get_deimos_continuum(spectra_sol)
print("Convolving Solar spectrum from %s..." % ArcturusFileName)
spectra_sol2 = convolve_spectrum(wavelength_arc,spectra_sol2)
print("Convolving Arcturus spectrum from %s..." % ArcturusFileName)
spectra_arc = convolve_spectrum(wavelength_arc,spectra_arc)



# save the convolved spectra and their labels
print("Saving Convolved Spectra to %s" % ('convolved_masking_spectra.npz'))
np.savez(OutputDir + 'convolved_masking_spectra.npz',
         spectra_sol = spectra_sol, spectra_sol2 = spectra_sol2, spectra_arc = spectra_arc,
         wavelength = wavelength_template)

print("Convolution completed!")
