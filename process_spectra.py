from __future__ import absolute_import, division, print_function
import numpy as np
from astropy.io import fits
import multiprocessing
import utils

'''
Code for reading in 1-d DEIMOS spectra reduced by the spec2d code:
http://deep.ps.uci.edu/spec2d/.

This code:
1) Reads in 1-d DEIMOS spectra .fits files and concatenates spectra from the
red and blue CCDs.
2) Interpolates spectra onto the standard wavelength template for DEIMOS.
3) Outputs all object spectra in a .npz file.
'''

# # # Settings # # #

'''
Which method of fitting the DEIMOS spectrum do you want to use:
'Horne' or 'Bxspf'
'''
method = 'Horne'

# Directory reduced DEIMOS spectra reside in
DEIMOSDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/'
InputDir = DEIMOSDir + 'U112/m15msk/'
# Input List of DEIMOS spectra to be processed for fitting
InputList = InputDir + 'spec1d.m15msk.txt'
# D-Payne Directory
D_PayneDir = DEIMOSDir + 'D-Payne/'
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
# Restore spectral mask from Kirby+ 2008
print('Restoring spectral mask from Kirby+ 2008...')
kirby_2008 = utils.get_spectral_mask_dict(name='kirby_2008')
mask = utils.generate_mask_from_dict(**kirby_2008)


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


# Initialize Arrays
# ObjNumber = np.empty(len(InputList), dtype='<U11')
# wavelength = np.empty((len(InputList), len(wavelength_template)))
# spec = np.empty((len(InputList), len(wavelength_template)))
# ivar = np.empty((len(InputList), len(wavelength_template)))


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
    print('Applying spectral mask for spectra #%s' % ObjNumber)
    spec_err[mask] = 999.
    print('Normalizing spectra #%s' % ObjNumber)
    cont_spec = \
        utils.get_deimos_continuum(spec, spec_err=spec_err,
                                   wavelength=wavelength,
                                   cont_pixels=cont_reg,
                                   wavelength_diff_matrix=wavelength_diff_matrix)
    spec = spec / cont_spec
    return(ObjNumber, wavelength, spec, ivar, RA, Dec)


print('Beginning processing of all spectra')
pool = multiprocessing.Pool(multiprocessing.cpu_count())
ObjNumber, wavelength, spec, ivar, RA, Dec \
    = pool.map(process_deimos_spectra, range(ObjList.shape[0]))
print('Completed processing of all spectra')

# Save processed spectra
print('Saving all processed spectra to %s' % OutputFile)
np.savez(OutputDir + OutputFile, obj=ObjNumber, wavelength=wavelength,
         spec=spec, spec_err=(ivar**-1), RA=RA, Dec=Dec)
