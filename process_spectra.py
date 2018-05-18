'''
Code for reading in 1-d DEIMOS spectra reduced by the spec2d code:
http://deep.ps.uci.edu/spec2d/.

This code:
1) Reads in 1-d DEIMOS spectra .fits files and concatenates spectra from the
red and blue CCDs.
2) Interpolates spectra onto the standard wavelength template for DEIMOS.
3) Outputs all object spectra in a .npz file.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import multiprocessing
import utils

# # # Settings # # #

'''
Which method of fitting the DEIMOS spectrum do you want to use:
'Horne' or 'Bxspf'
'''
method = 'Horne'

# Directory reduced DEIMOS spectra reside in
# ===================================================================================
'''
# Nathan's Laptop
DEIMOSDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/'
InputDir = DEIMOSDir + 'U112/m15msk/'
'''
# BRC
DEIMOSDir = '/global/scratch/nathan_sandford/DEIMOS/'
InputDir = DEIMOSDir + 'U112/m15msk/m15msk/'
# '''
# ===================================================================================
# Input List of DEIMOS spectra to be processed for fitting
InputList = InputDir + 'spec1d.m15msk.txt'
# D-Payne Directory
D_PayneDir = utils.D_PayneDir
# Output directory for processed spectra
OutputDir = D_PayneDir + 'spectra/M15/'
# File containing processed spectra
OutputFile = 'm15_'+method+'.npz'

# Extract object names from input list
print('Restoring list of objects...')
temp = open(InputList, 'r')
ObjList = temp.read().split('\n')[:-2]
temp.close

# Restore Wavelength Template
print('Restoring Wavelength Template...')
wavelength_template = utils.load_wavelength_array()

# Calculate matrix of distances between wavelengths
print('Calculating matrix of distances between wavelengths...')
wavelength_diff_matrix \
    = wavelength_template[:, np.newaxis] - wavelength_template

# Restore DEIMOS continuum pixels
print('Restoring DEIMOS continuum pixels...')
cont_reg = utils.load_deimos_cont_pixels()

# Restore telluric mask from Kirby+ 2008
print('Restoring telluric mask from Kirby+ 2008...')
kirby_2008_telluric = utils.get_spectral_mask_dict(name='kirby_2008_telluric')
mask = utils.generate_mask_from_dict(**kirby_2008_telluric)

# Restore spectral template of typical RGB star
print('Restoring spectral template...')
temp = np.load(D_PayneDir + '/other_data/typical_RGB_spectra.npz')
template_spec = temp['spec'][(wavelength_template > 8450) &
                             (wavelength_template < 8700)]
temp.close()

# Prepping cross-correlation function
print('Prepping cross-correlation calculation...')
wavelength_CaIItriplet = wavelength_template[(wavelength_template > 8450) &
                                             (wavelength_template < 8700)]
dv_grid = np.linspace(-300, 300, 300)
template_grid = np.empty((len(dv_grid), len(wavelength_CaIItriplet)))
for i, dv in enumerate(dv_grid):
    template_grid[i] = utils.doppler_shift(wavelength=wavelength_CaIItriplet,
                                           flux=template_spec, dv=dv)


def get_deimos_spectra(Obj, method, InputDir=None):
    '''
    Read in and concatenate red and blue spectra
    '''
    ObjHDUL = fits.open(InputDir + Obj)

    waveB = ObjHDUL[method+'-B'].data['LAMBDA'][0]
    waveR = ObjHDUL[method+'-R'].data['LAMBDA'][0]
    wave = np.concatenate((waveB, waveR))

    specB = ObjHDUL[method+'-B'].data['SPEC'][0]
    specR = ObjHDUL[method+'-R'].data['SPEC'][0]
    spec = np.concatenate((specB, specR))

    ivarB = ObjHDUL[method+'-B'].data['IVAR'][0]
    ivarR = ObjHDUL[method+'-R'].data['IVAR'][0]
    ivarB = 1e-16 * np.ones(len(ivarB))  # completely ignore blue channel
    ivar = np.concatenate((ivarB, ivarR))
    ivar[ivar == 0] = 1e-16  # Avoid np.inf in spec_err

    RA = ObjHDUL[method+'-B'].header['RA_OBJ']
    Dec = ObjHDUL[method+'-B'].header['DEC_OBJ']
    return (wave, spec, ivar, RA, Dec)


def interpolate_deimos_spectra(wave, spec, spec_err):
    '''
    Interpolates a DEIMOS spectrum onto the default wavelength grid
    '''
    if len(wave) != 16250:
        spec = np.interp(wavelength_template, wave, spec)
        spec_err = np.interp(wavelength_template, wave, spec_err)
        wave = np.copy(wavelength_template)
    return(wave, spec, spec_err)


def fast_RV(spec, spec_err, plot=False):
    '''
    Quick radial velocity determination by cross-correlating observed spectrum
    with a template spectrum in the region around the Ca II triplet.
    '''

    # Consider only region around Ca II triplet
    temp_spec = spec[(wavelength_template > 8450) &
                     (wavelength_template < 8700)]
    temp_spec_err = spec_err[(wavelength_template > 8450) &
                             (wavelength_template < 8700)]

    # Cross-Correlate
    num = np.sum(template_grid * temp_spec / temp_spec_err**2, axis=1)
    den = np.sum(template_grid * template_grid / temp_spec_err**2, axis=1)
    xcorr = num / den

    # Naively take maximum of CCF
    dv = dv_grid[np.argmax(xcorr)]

    # Plot output
    if plot:
        plt.plot(dv_grid, xcorr)
        plt.xlabel('RV (km/s)')
        plt.ylabel('Cross-Corellation')
        plt.show()

    return(dv)


def process_deimos_spectra(i):
    '''
    Processes all spectra in InputList
    '''
    Obj = ObjList[i]
    ObjNumber = Obj[14:-5]
    print('Processing spectra for object: %s' % Obj)

    print('Restoring spectra #%s' % ObjNumber)
    wave_temp, spec_temp, ivar_temp, RA, Dec = \
        get_deimos_spectra(Obj=Obj, method=method, InputDir=InputDir)

    print('Interpolating spectra #%s' % ObjNumber)
    wavelength, spec, spec_err = \
        interpolate_deimos_spectra(wave=wave_temp,
                                   spec=spec_temp,
                                   spec_err=(ivar_temp**-1))

    print('Applying telluric mask for spectra %s' % ObjNumber)
    spec_err[mask] = 1e16

    print('Finding radial velocity from CCF for spectra #%s' % ObjNumber)
    dv = fast_RV(spec, spec_err, plot=False)
    print('%s has Radial Velocity = %.0f' % (ObjNumber, dv))
    print('Shifting spectra %s to rest frame' % ObjNumber)
    spec = utils.doppler_shift(wavelength=wavelength, flux=spec, dv=-dv)

    print('Normalizing spectra %s' % ObjNumber)
    cont_spec = \
        utils.get_deimos_continuum(spec, spec_err=spec_err,
                                   wavelength=wavelength,
                                   cont_pixels=cont_reg,
                                   wavelength_diff_matrix=wavelength_diff_matrix)
    spec = spec / cont_spec

    # Handle regions where continuum is zero
    spec_err[np.isnan(spec)] = 1e16
    spec[np.isnan(spec)] = 0

    return(ObjNumber, wavelength, spec, spec_err, dv, RA, Dec)


print('Beginning processing of all spectra')
pool = multiprocessing.Pool(multiprocessing.cpu_count())
temp = pool.map(process_deimos_spectra, range(len(ObjList)))
temp = list(zip(*temp))
ObjNumber, wavelength, spec, spec_err, dv, RA, Dec = temp
print('Completed processing of all spectra')

# Save processed spectra
print('Saving all processed spectra to %s' % OutputFile)
np.savez(OutputDir + OutputFile, obj=ObjNumber, wavelength=wavelength,
         spec=spec, spec_err=spec_err, dv=dv, RA=RA, Dec=Dec)
