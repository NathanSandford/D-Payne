# code for fitting spectra, using the models in model_spectra.py
# Adapted from Kareem's binspec/fitting.py and Yuan-Sen's test_NN.py

# import packages
# python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
import model_spectra
import utils

# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_deimos_cont_pixels()


def fit_normalized_spectrum_single_star_model(norm_spec, spec_err,
                                              NN_coeffs, p0=None, num_p0=1):
    '''
    fit a single-star model to a single combined spectrum

    p0 is an initial guess for where to initialize the optimizer. Because
    this is a simple model, having a good initial guess is usually not
    important.

    if num_p0 is set to a number greater than 1, this will initialize a bunch
    of different walkers at different points in parameter space. If they
    converge on different solutions, it will pick the one with the lowest
    chi2.
    labels = [[alpha/Fe],[Mg/Fe],[Si/Fe],[S/Fe],[Ar/Fe],[Ca/Fe],[Ti/Fe],
                [Fe/H],Teff,logg,dv]

    returns:
    popt: the best-fit labels
    pcov: the covariance matrix,
          from which you can get formal fitting uncertainties
    model_spec: the model spectrum corresponding to popt
    '''

    tol = 5e-4  # tolerance for when the optimizer should stop optimizing.

    def fit_func(dummy_variable, *labels):
        norm_spec = model_spectra.get_spectrum_from_neural_net(
                                                        labels=labels,
                                                        NN_coeffs=NN_coeffs)
        return(norm_spec)

    '''
    If no initial guess is supplied,
    start with labels for a 'typical' RGB star.
    '''
    if p0 is None:
        p0 = [0, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 4100, 0.5, 0]

    # don't allow the minimimizer outside Teff = [3000, 10000], etc.
    bounds = [[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
               3000, 0.0, -200],
              [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5,
               10000, 5.0, 200]]

    '''
    If we want to initialize many walkers
    in different parts of parameter space, do so now.
    '''
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0=p0,
                                                               bounds=bounds,
                                                               num_p0=num_p0,
                                                               vrange=10)

    # run the optimizer
    popt, pcov, model_spec = fit_all_p0s(fit_func=fit_func,
                                         norm_spec=norm_spec,
                                         spec_err=spec_err,
                                         all_x0=all_x0,
                                         bounds=bounds,
                                         tol=tol)
    return(popt, pcov, model_spec)


def generate_starting_guesses_to_initialze_optimizers(p0, bounds,
                                                      num_p0, vrange=10):
    '''
    If we want to initialize many walkers in different parts of parameter space

    p0 is the initial guess around which to cluster our other guesses
    bounds is the region of which parameter space the optimizer is allowed to
    explore.
    num_p0 is how many walkers we want.
    vrange is half the range in velocity that the starting guesses should be
    spread over.

    If you're fitting a close binary with very large velocity offset, it can
    occur that the walkers are initialized too far from the best-fit velocity
    to find it. For such cases, it can be useful to increase vrange to ~50.
    '''
    all_x0 = [p0]
    lower, upper = bounds

    if num_p0 > 1:
        for i in range(num_p0-1):
            alpha = np.random.uniform(max(lower[0], p0[0] - 0.05),
                                      min(upper[0], p0[0] + 0.05))
            Mg = np.random.uniform(max(lower[1], p0[1] - 0.05),
                                   min(upper[1], p0[1] + 0.05))
            Si = np.random.uniform(max(lower[2], p0[2] - 0.05),
                                   min(upper[2], p0[2] + 0.05))
            S = np.random.uniform(max(lower[3], p0[3] - 0.05),
                                  min(upper[3], p0[3] + 0.05))
            Ar = np.random.uniform(max(lower[4], p0[4] - 0.05),
                                   min(upper[4], p0[4] + 0.05))
            Ca = np.random.uniform(max(lower[5], p0[5] - 0.05),
                                   min(upper[5], p0[5] + 0.05))
            Ti = np.random.uniform(max(lower[6], p0[6] - 0.05),
                                   min(upper[6], p0[6] + 0.05))
            feh = np.random.uniform(max(lower[7], p0[7] - 0.2),
                                    min(upper[7], p0[7] + 0.2))
            teff = np.random.uniform(max(lower[8], p0[8] - 300),
                                     min(upper[8], p0[8] + 500))
            logg = np.random.uniform(max(lower[9], p0[9] - 0.2),
                                     min(upper[9], p0[9] + 0.2))
            dv = np.random.uniform(max(lower[10], p0[10] - vrange),
                                   min(upper[10], p0[10] + vrange))
            this_p0 = np.array([alpha, Mg, Si, S, Ar, Ca, Ti, feh,
                                teff, logg, dv])
            all_x0.append(this_p0)

    '''
    Make sure none of these walkers got initialized outside the allowed regions
    of label space. This should not happen unless p0 was bad.
    '''
    for j, p0 in enumerate(all_x0):
        for i, p in enumerate(p0):
            if (p < bounds[0][i]):
                all_x0[j][i] = bounds[0][i] + 1e-5
            if (p > bounds[1][i]):
                all_x0[j][i] = bounds[1][i] - 1e-5
    return(all_x0)


def fit_all_p0s(fit_func, norm_spec, spec_err, all_x0, bounds, tol=5e-4):
    '''
        Loop through all the points to initialize the optimizer.
        If there are more than one, run the optimizer at each point
        sequentially and choose the best model as the one that minimizes chi2.
        fit_func is the function to predict the spectrum for a given model.
        '''
    from scipy.optimize import curve_fit
    all_popt, all_chi2, all_model_specs, all_pcov = [], [], [], []
    for i, x0 in enumerate(all_x0):
        try:
            popt, pcov = curve_fit(fit_func, xdata=[], ydata=norm_spec,
                                   sigma=spec_err, p0=x0,
                                   bounds=bounds, ftol=tol, xtol=tol,
                                   absolute_sigma=True, method='trf')
            model_spec = fit_func([], *popt)
            chi2 = np.sum((model_spec - norm_spec)**2/spec_err**2)
        # failed to converge (should not happen for a simple model)
        except RuntimeError:
            popt, pcov = x0, np.zeros((len(x0), len(x0)))
            model_spec = np.copy(norm_spec)
            chi2 = np.inf
        all_popt.append(popt)
        all_chi2.append(chi2)
        all_model_specs.append(model_spec)
        all_pcov.append(pcov)
    all_popt, all_chi2, all_model_specs, all_pcov = np.array(all_popt), \
        np.array(all_chi2), np.array(all_model_specs), np.array(all_pcov)

    best = np.argmin(all_chi2)
    popt, pcov, model_spec = all_popt[best], all_pcov[best], \
        all_model_specs[best]
    return(popt, pcov, model_spec)
