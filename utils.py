'''
A few low-level functions that are used throughout.
(Adapted from Kareem's binspec/utils.py.)
'''

# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import interpolate

'''
Important Directories
'''
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'
KuruczSpectraDir \
    = '/global/home/users/nathan_sandford/kurucz/synthetic_spectra/'


def read_in_neural_network(name='norm_spectra_approx'):
    '''
    Read in the weights and biases parameterizing a particular neural network.
    Looks for NN named: D_PayneDir + 'neural_nets/' + name + '.npz'
    '''
    if name == 'norm_spectra_approx':
        path = D_PayneDir+'neural_nets/NN_norm_spectra_approx.npz'
    elif name == 'norm_spectra_true':
        path = D_PayneDir+'neural_nets/NN_norm_spectra_true.npz'
    else:
        path = D_PayneDir + 'neural_nets/' + name + '.npz'
    tmp = np.load(path)
    try:  # Try Loading in NN with two hidden layers
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2,
                     b_array_0, b_array_1, b_array_2,
                     x_min, x_max)
    except KeyError:  # Try Loading in NN with one hidden layers
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1,
                     b_array_0, b_array_1,
                     x_min, x_max)
    tmp.close()
    return NN_coeffs


def load_wavelength_array():
    '''
    Read in the default wavelength grid onto which we interpolate all spectra.
    Template created with:
        D_PayneDir + 'other_data/create_deimos_wavelength_template.py'
    '''
    path = D_PayneDir + 'other_data/deimos_wavelength.npz'
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength


def doppler_shift(wavelength, flux, RV):
    '''
        RV is the radial velocity in km/s.
        Positive RV means the object is moving away.

        This linear interpolation is actually not that accurate, but is fine if
        you only care about accuracy to the level of a few tenths of a km/s.
        If you care about better accuracy, you can do better with spline
        interpolation.
        '''
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - RV/c)/(1 + RV/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return(new_flux)


def get_continuum_pixels(wavelength, normalized_spectra, wavelength_template,
                         cut=0.96):
    '''
    Finds continuum pixels on the standard wavelength grid by identifying
    continuum regions in a theoretically normalized spectrum.
    '''

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
    Reads in the default list of DEIMOS pixels to use for continuum fitting.

    Default list created with:
        D_PayneDir + 'other_data/determine_continuum_regions.py'
    '''
    path = D_PayneDir + 'other_data/deimos_cont_pixels.npy'
    cont_pixels = np.load(path)
    return cont_pixels


def get_deimos_continuum_poly(spec, spec_err=None,
                              wavelength=None,
                              cont_pixels=None, bounds=[6250, 9500],
                              deg=6, verbose=False):
    '''
    Approximate continuum using a polynomial
    '''
    # Load standard DEIMOS wavelength grid
    if wavelength is None:
        if verbose:
            print('Loading wavelength grid...')
        wavelength = load_wavelength_array()
    # If no error given, assume 1 everywhere
    if spec_err is None:
        if verbose:
            print('No errors given, assuming all are equal')
        spec_err = np.ones(len(wavelength))
    # Limit to continuum regions
    if cont_pixels is not None:
        wave = wavelength[cont_pixels]
        spec = spec[cont_pixels]
        spec_err = spec_err[cont_pixels]
    else:
        wave = wavelength
    # Fitting regions
    fit_reg = (wave > bounds[0]) & (wave < bounds[1]) \
        & (spec > 0)
    # Calculate Continuum
    cont = np.polyfit(wave[fit_reg], spec[fit_reg],
                      deg=deg, w=np.sqrt(spec_err[fit_reg])**-1)
    cont = np.poly1d(cont)
    return(cont(wavelength))


def get_deimos_continuum_smooth(spec, spec_err=None, wavelength=None,
                                cont_pixels=None,
                                wavelength_diff_matrix=None,
                                verbose=False):
    '''
    Approximate continuum as a smoothed version of the spectrum using only
    continuum regions. This is modeled after the method used in
    Kirby et al. (2008).
    '''
    # Load standard DEIMOS wavelength grid
    if wavelength is None:
        if verbose:
            print('Loading wavelength grid...')
        wavelength = load_wavelength_array()
    # If no error given, assume 1 everywhere
    if spec_err is None:
        if verbose:
            print('No errors given, assuming all are equal')
        spec_err = np.ones(len(wavelength))
    # Load continuum regions
    if cont_pixels is None:
        if verbose:
            print('Loading continuum regions...')
        cont_pixels = load_deimos_cont_pixels()
    m = np.zeros(len(wavelength))
    m[cont_pixels] = 1
    # Load / Calculate matix of distances between wavelengths
    if wavelength_diff_matrix is None:
            if verbose:
                print('Calculating wavelength difference matrix...')
            wavelength_diff_matrix = wavelength[:, np.newaxis] - wavelength
    # Calculates weights / smoothing kernel
    if verbose:
        print('Calculating weights...')
    w = m / spec_err * np.exp(-np.power(wavelength_diff_matrix / 10, 2) / 2)
    # Continuum Spectrum (Kirby+ 2008 Eq. 2)
    if verbose:
        print('Calculating continuum...')
    num = np.sum(spec * w, axis=1)
    den = np.sum(w, axis=1)
    cont_spec = num / den
    if verbose:
        print('Continuum calculation complete!')
    return(cont_spec)


def get_spectral_mask_dict(name='kirby_2008'):
    '''
    Reads in dictionary of specific lines to mask and the spectral range over
    which the mask should be applied.
    '''
    if name == 'kirby_2008':
        mask_dict = {'B band': [6864, 7020],
                     'A band': [7591, 7703],
                     'strong telluric asorption': [8225, 8265],
                     'Ca I 6343': [6341, 6346],
                     'Ca I 6362': [6356, 6365],
                     'H alpha': [6559.797, 6565.797],
                     'K I 7665': [7662, 7668],
                     'V I 8116,8119 hyperfine structure': [8113, 8123],
                     'poorly modeled absorption in Arcturus': [8317, 8330],
                     'Ca II 8498': [8488.023, 8508.023],
                     'Ca II 8542': [8525.091, 8561.091],
                     'Ca II 8662': [8645.141, 8679.141],
                     'Mg I 8807': [8804.756, 8809.756]}
        return (mask_dict)
    elif name == 'kirby_2008_telluric':
        mask_dict = {'B band': [6864, 7020],
                     'A band': [7591, 7703],
                     'strong telluric asorption': [8225, 8265]}
        return(mask_dict)
    elif name == 'kirby_2008_stellar':
        mask_dict = {'Ca I 6343': [6341, 6346],
                     'Ca I 6362': [6356, 6365],
                     'H alpha': [6559.797, 6565.797],
                     'K I 7665': [7662, 7668],
                     'V I 8116,8119 hyperfine structure': [8113, 8123],
                     'poorly modeled absorption in Arcturus': [8317, 8330],
                     'Ca II 8498': [8488.023, 8508.023],
                     'Ca II 8542': [8525.091, 8561.091],
                     'Ca II 8662': [8645.141, 8679.141],
                     'Mg I 8807': [8804.756, 8809.756]}
        return(mask_dict)
    else:
        print('No mask named %s' % name)


def generate_mask_from_dict(**kwargs):
    '''
    Generates a mask on the standard wavelength template
    from a masking dictionary.
    '''
    wavelength = load_wavelength_array()
    mask = np.array([])
    for key in kwargs:
        lower, upper = kwargs[key]
        temp_mask = np.where((wavelength > lower) & (wavelength < upper))
        mask = np.append(mask, temp_mask)
    return(mask.astype(int))


def restore_mask_from_file(name):
    '''
    Restores mask from file.
    '''
    if name[-4:] != '.npy':
        filename = name + '.npy'
    else:
        filename = name

    if name[:4] != 'mask':
        filename = 'mask.' + filename
    else:
        filename = filename

    try:
        mask = np.load(D_PayneDir+'other_data/'+filename)
        return(mask)
    except FileNotFoundError:
        print('No mask named %s' % name)


def get_chi2_difference(norm_spec, spec_err, norm_model_A, norm_model_B):
    '''
    For model comparison. Returns chi2_modelA - chi2_modelB.
    norm_model_A & B are normalized spectra predicted by two different models.
    So e.g., if model A is more simple than model B (say, a single-star
        vs a binary model), one would expect this to be positive.
    '''
    chi2_A = np.sum((norm_spec - norm_model_A)**2 / spec_err**2)
    chi2_B = np.sum((norm_spec - norm_model_B)**2 / spec_err**2)
    return chi2_A - chi2_B
