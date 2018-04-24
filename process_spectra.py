from __future__ import absolute_import, division, print_function
import numpy as np
from astropy.io import fits

'''
Code for reading in 1-d DEIMOS spectra reduced by the spec2d code: http://deep.ps.uci.edu/spec2d/.
This code:
1) Reads in 1-d DEIMOS spectra .fits files and concatenates spectra from the red and blue CCDs.
2) Interpolates spectra onto the standard wavelength template for DEIMOS.
3) Outputs all object spectra in a .npz file.
'''

### Settings ###
# Directory reduced DEIMOS spectra reside in
DEIMOSDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/'
InputDir = DEIMOSDir + 'U112/m15msk/'
# List of DEIMOS spectra to be processed for fitting
InputList = InputDir + 'spec1d.m15msk.txt'
# D-Payne Directory
D_PayneDir = DEIMOSDir + 'D-Payne/'
# Output directory for processed spectra
OutputDir = D_PayneDir + 'spectra/M15/'
# File containing processed spectra
OutputFile = 'm15_unnormalized_'+method+'.npz'

#Which method of fitting the DEIMOS spectrum do you want to use ('Horne' or 'Bxspf')
method = 'Horne'

def get_deimos_spectra(Obj,method,InputDir = None):
    '''
    Read in and concatenate red and blue spectra
    '''
    ObjHDUL = fits.open(InputDir + Obj)
    
    waveB = ObjHDUL[method+'-B'].data['LAMBDA'][0]
    waveR = ObjHDUL[method+'-R'].data['LAMBDA'][0]
    wave = np.concatenate((waveB,waveR))
    
    specB = ObjHDUL[method+'-B'].data['SPEC'][0]
    specR = ObjHDUL[method+'-R'].data['SPEC'][0]
    spec = np.concatenate((specB,specR))
    
    ivarB = ObjHDUL[method+'-B'].data['IVAR'][0]
    ivarR = ObjHDUL[method+'-R'].data['IVAR'][0]
    ivar = np.concatenate((ivarB,ivarR))
    
    RA = ObjHDUL[method+'-B'].header['RA_OBJ']
    Dec = ObjHDUL[method+'-B'].header['DEC_OBJ']
    return (wave,spec,ivar,RA,Dec)

def interpolate_deimos_spectra(wave,spec,spec_err):
    '''
    Interpolates a DEIMOS spectrum onto the default wavelength grid
    '''
    if len(wave) != 16250:
        standard_grid = load_wavelength_array()
        spec = np.interp(standard_grid, wave, spec)
        spec_err = np.interp(standard_grid, wave, spec_err)
        wave = np.copy(standard_grid)
    return(wave,spec,spec_err)


# Initialize Arrays
ObjNumber = np.empty(len(ObjList),dtype='<U11')
wavelength = np.empty((len(ObjList),len(wavelength_template)))
spec = np.empty((len(ObjList),len(wavelength_template)))
ivar = np.empty((len(ObjList),len(wavelength_template)))

# Process all spectra
for i,Obj in enumerate(ObjList):
    ObjNumber[i] = Obj[14:-5]
    wave_temp,spec_temp,ivar_temp,RA,Dec = get_deimos_spectra(Obj=Obj,method=method,InputDir=InputDir)
    wavelength[i],spec[i],ivar[i] = interpolate_deimos_spectra(wave=wave_temp,spec=spec_temp, spec_err=ivar_temp)

# Save processed spectra
np.savez(OutputDir+OutputFile,obj=ObjNumber,wavelength=wavelength,spec=spec,spec_err=ivar,RA=RA,Dec=Dec)

