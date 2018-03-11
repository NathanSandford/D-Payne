'''
A few low-level functions that are used throughout.
Adapted from Kareem's binspec/utils.py.
'''
    
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np

D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne/'

def read_in_neural_network(name = 'normalized_spectra'):
    '''
    read in the weights and biases parameterizing a particular neural network. 
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in. 
    '''
    if name == 'normalized_spectra':
        path = 'neural_nets/NN_normalized_spectra.npz'
    elif name == 'unnormalized_spectra':
        path = 'neural_nets/NN_unnormalized_spectra.npz'
    elif name == 'radius':
        path = 'neural_nets/NN_radius.npz'
    elif name == 'Teff2_logg2':
        path = 'neural_nets/NN_Teff2_logg2.npz'
    tmp = np.load(path)
    
    # some of the networks we train have one hidden layer; others have two. 
    # assume the one we're looking for has two; if it doesn't, we won't find 
    # w_array_2 and b_array_2. 
    try:
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    except KeyError:
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max)
    tmp.close()
    return NN_coeffs

def load_wavelength_array():
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    '''
    path = D_PayneDir + 'other_data/deimos_wavelength.npz'
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength

def load_wavelength_diff_matrix():
    '''
    read in matrix of distances between points in wavelength grid
    '''
    path = D_PayneDir + 'other_data/deimos_wavelength_diff.npz'
    tmp = np.load(path)
    wavelength_diff_matrix = tmp['wavelength_diff']
    tmp.close()
    return wavelength_diff_matrix

def interpolate_deimos_spectra(wave,spec,spec_err):
    '''
    interpolates a DEIMOS spectrum onto the default wavelength grid
    '''
    if len(wave) != 16250:
        print('fixing wavelength...')
        standard_grid = utils.load_wavelength_array()
        spec = np.interp(standard_grid, wave, spec)
        spec_err = np.interp(standard_grid, wave, spec_err)
        wave = np.copy(standard_grid)
    return(wave,spec,specerr)

def load_deimos_cont_pixels():
    '''
    read in the default list of APOGEE pixels to use for continuum fitting.
    These are taken from Melissa Ness' work with the Cannon
    '''
    path = D_PayneDir + 'other_data/deimos_cont_pixels.npz'
    tmp = np.load(path)
    cont_pixels = tmp['cont_pixels']
    tmp.close()
    return cont_pixels

def doppler_shift(wavelength, flux, dv):
    '''
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.
    
    This linear interpolation is actually not that accurate, but is fine if you 
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation. 
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c)) 
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux

def get_deimos_continuum(spec, spec_err=None, wavelength = None,
                         cont_pixels = None,
                         wavelength_diff_matrix = None):
    '''
    Approximate continuum as a smoothed version of the spectrum using only
    continuum regions. This is modeled after the method used in Kirby et al. (2008).
    '''
    
    # Load standard DEIMOS wavelength grid
    if wavelength is None:
        print('Loading wavelength grid...')
        wavelength = utils.load_wavelength_array()

    # If no error given, assume 1 everywhere
    if spec_err is None:
        print('No errors given, assuming all are equal')
        spec_err = np.ones(len(wavelength))

    # Load continuum regions
    if cont_pixels is None:
        print('Loading continuum regions...')
        cont_pixels = utils.load_deimos_cont_pixels()
    m = np.zeros(len(wavelength))
    m[cont_pixels] = 1

    # Load / Calculate matix of distances between wavelengths
    if wavelength_diff_matrix is None:
        try:
            print('Loading wavelength difference matrix...')
            wavelength_diff_matrix = utils.load_wavelength_diff_matrix()
        except FileNotFoundError:
            print('No wavelength difference matrix found.')
            print('Calculating wavelength difference matrix...')
            wavelength_diff_matrix = wavelength[:,np.newaxis] - wavelength

    # Calculates weights / smoothing kernel
    print('Calculating weights...')
    w = m * spec_err * np.exp(-np.power(wavelength_diff_matrix/10,2) / 2 )

    # Continuum Spectrum (Kirby+ 2008 Eq. 2)
    print('Calculating continuum...')
    num = np.sum(spec * w,axis=1)
    den = np.sum(w,axis=1)
    cont_spec = num/den
    
    print('Continuum calculation complete!')
    return cont_spec

def get_chi2_difference(norm_spec, spec_err, norm_model_A, norm_model_B):
    '''
    for model comparison. Returns chi2_modelA - chi2_modelB.
    norm_model_A & B are normalized spectra predicted by two different models. 
    So e.g., if model A is more simple than model B (say, a single-star 
        vs a binary model), one would expect this to be positive. 
    '''
    chi2_A = np.sum((norm_spec - norm_model_A)**2/spec_err**2)
    chi2_B = np.sum((norm_spec - norm_model_B)**2/spec_err**2)
    return chi2_A - chi2_B
