'''
Adapted from Kareem's binspec/model_spectra.py.

Code for predicting the spectrum of a single star in normalized space.
'''
# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
import utils

# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_deimos_cont_pixels()


def sigmoid(z):
    '''
    This is the activation function used by default in all our neural
    networks.
    '''
    return 1.0/(1.0 + np.exp(-z))


def get_spectrum_from_neural_net(labels, NN_coeffs):
    '''
    Predict the rest-frame normalized spectrum of a single star.
    '''

    # Assuming your NN had two hidden layer:
    try:
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, \
            x_min, x_max = NN_coeffs
        n_hidden = 2
    # If it instead had one hidden layer:
    except ValueError:
        w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max = NN_coeffs
        n_hidden = 1
    scaled_labels = (labels - x_min)/(x_max - x_min) - 0.5

    if n_hidden == 1:
        '''
        This is just efficient matrix multiplication---
        quite a bit faster than np.dot().
        '''
        inside = np.einsum('ijk,k->ij', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,ij->i', w_array_1, sigmoid(inside)) + b_array_1
    elif n_hidden == 2:
        '''
        I'm not familiar enough with einsum to do the same for NN with 2
        hidden layers
        '''
        inside = sigmoid(np.dot(w_array_0, scaled_labels) + b_array_0)
        middle = sigmoid(np.sum(w_array_1 * inside, axis=1) + b_array_1)
        outside = w_array_2 * middle + b_array_2
    spectrum = outside

    return spectrum
