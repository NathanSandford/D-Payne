'''
A few low-level functions that are used throughout.
(Adapted from Kareem's binspec/utils.py.)
'''

# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import interpolate

# D_PayneDir = '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne/'
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'


def read_in_neural_network(name='norm_spectra_approx'):
    '''
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    '''
    if name == 'norm_spectra_approx':
        path = D_PayneDir+'neural_nets/NN_norm_spectra_approx.npz'
    elif name == 'norm_spectra_true':
        path = D_PayneDir+'neural_nets/NN_norm_spectra_true.npz'
    else:
        path = name
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
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1,
                     b_array_2, x_min, x_max)
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


def doppler_shift(wavelength, flux, dv):
    '''
        dv is the radial velocity in km/s.
        Positive dv means the object is moving away.

        This linear interpolation is actually not that accurate, but is fine if
        you only care about accuracy to the level of a few tenths of a km/s.
        If you care about better accuracy, you can do better with spline
        interpolation.
        '''
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux


def get_continuum_pixels(wavelength, normalized_spectra, cut=0.96):
    '''
    Finds continuum pixels on the standard wavelength grid by identifying
    continuum regions in a normalized spectrum.
    '''
    # Load standard DEIMOS wavelength grid
    wavelength_template = load_wavelength_array()

    # Identify continuum region in normalized spectrum
    temp = np.zeros(len(wavelength))
    temp[normalized_spectra > cut] = 1
    n = np.zeros(len(wavelength))
    for i, item in enumerate(wavelength):
        n[i] = np.floor(np.average(temp[(wavelength > item - 0.25) &
                                        (wavelength < item + 0.25)]))
    cont_reg = interpolate.interp1d(wavelength, n)

    # Interpolate continuum regions onto standard wavelength grid
    m = np.floor(cont_reg(wavelength_template))
    cont_pixels = np.argwhere(m)
    return(cont_pixels)


def load_deimos_cont_pixels():
    '''
    read in the default list of DEIMOS pixels to use for continuum fitting.
    '''
    path = D_PayneDir + 'other_data/deimos_cont_pixels.npy'
    cont_pixels = np.load(path)
    return cont_pixels


def get_deimos_continuum(spec, spec_err=None, wavelength=None,
                         cont_pixels=None,
                         wavelength_diff_matrix=None):
    '''
    Approximate continuum as a smoothed version of the spectrum using only
    continuum regions. This is modeled after the method used in
    Kirby et al. (2008).
    '''

    # Load standard DEIMOS wavelength grid
    if wavelength is None:
        print('Loading wavelength grid...')
        wavelength = load_wavelength_array()

    # If no error given, assume 1 everywhere
    if spec_err is None:
        print('No errors given, assuming all are equal')
        spec_err = np.ones(len(wavelength))

    # Load continuum regions
    if cont_pixels is None:
        print('Loading continuum regions...')
        cont_pixels = load_deimos_cont_pixels()
    m = np.zeros(len(wavelength))
    m[cont_pixels] = 1

    # Load / Calculate matix of distances between wavelengths
    if wavelength_diff_matrix is None:
        try:
            print('Loading wavelength difference matrix...')
            wavelength_diff_matrix = load_wavelength_diff_matrix()
        except FileNotFoundError:
            print('No wavelength difference matrix found.')
            print('Calculating wavelength difference matrix...')
            wavelength_diff_matrix = wavelength[:, np.newaxis] - wavelength

    # Calculates weights / smoothing kernel
    print('Calculating weights...')
    w = m * spec_err * np.exp(-np.power(wavelength_diff_matrix/10, 2) / 2)

    # Continuum Spectrum (Kirby+ 2008 Eq. 2)
    print('Calculating continuum...')
    num = np.sum(spec * w, axis=1)
    den = np.sum(w, axis=1)
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
