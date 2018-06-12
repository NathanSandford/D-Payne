'''
Code for predicting the normalized spectrum of a single star given labels.

Adapted from Kareem's binspec/model_spectra.py.
'''

# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
import utils
# Restore DEIMOS Wavelength Grid
wavelength = utils.load_wavelength_array()


def sigmoid(z):
    '''
    This is the activation function used by default in our neural networks.
    '''
    return 1.0/(1.0 + np.exp(-z))


def get_spectrum_from_neural_net(labels, NN_coeffs):
    '''
    Predict the rest-frame normalized spectrum of a single star.
    '''

    # Extract radial velocity from labels
    RV = labels[-1]
    labels = labels[:-1]

    try:  # Try Loading in NN with two hidden layers
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, \
            x_min, x_max = NN_coeffs
        n_hidden = 2
    except ValueError:  # Try Loading in NN with one hidden layers
        w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max = NN_coeffs
        n_hidden = 1
    scaled_labels = (labels - x_min)/(x_max - x_min) - 0.5

    if n_hidden == 1:
        '''
        This is just efficient matrix multiplication--
        quite a bit faster than np.dot().
        '''
        inside = np.einsum('ijk,k->ij', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,ij->i', w_array_1, sigmoid(inside)) + b_array_1
    elif n_hidden == 2:
        '''
        Almost twice as fast as using np.dot() and np.sum()
        '''
        inside = sigmoid(np.einsum('ijk,k->ij', w_array_0, scaled_labels)
                         + b_array_0)
        middle = sigmoid(np.einsum('ij->i', w_array_1 * inside) + b_array_1)
        outside = w_array_2 * middle + b_array_2
    spectrum = outside

    # Doppler shift spectrum
    spectrum = utils.doppler_shift(wavelength=wavelength,
                                   flux=spectrum, RV=RV)

    return(spectrum)
