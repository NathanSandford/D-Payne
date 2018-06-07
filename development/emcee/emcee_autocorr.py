import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee
import multiprocessing
from multiprocessing import Pool
import sys
import os

import utils
import model_spectra as NN
import fitting

# Number of steps to iterate
nsteps = int(sys.argv[1])

# read in the standard wavelength grid onto which we interpolate spectra.
wavelength = utils.load_wavelength_array()

# read in all individual neural networks we'll need.
NN_coeffs = utils.read_in_neural_network(name='norm_spectra_approx')

# Restore Observed spectra
D_PayneDir = utils.D_PayneDir
SpectraDir = D_PayneDir + 'spectra/obs_spectra/'
SpectraFile = 'm15_Horne.npz'
temp = np.load(SpectraDir + SpectraFile)
obj = temp['obj']
norm_spectra = temp['norm_spec']
full_spectra = temp['spec']
spectral_err = temp['spec_err']
dv = temp['dv']
RA_Dec = SkyCoord(temp['RA_Dec'])
temp.close()

# Standard Stars
spec_ind = 8
full_spec = full_spectra[spec_ind]
norm_spec = norm_spectra[spec_ind]
spec_err = spectral_err[spec_ind]

# kirby_2008_stellar = utils.get_spectral_mask_dict(name='kirby_2008_stellar')
mask = utils.generate_mask_from_file(name='008.0010337')
spec_err[mask] = 1e16
masked_wavelength = wavelength[mask]

matches = [8, 26]
feh = [0.33, -1.26]
alpha = [0.23, -0.36]
Teff = [5663.6, 5650.5]
logg = [4.3, 3.64]
dv = [0.0, 0.0]

i = 0
alphafe = alpha[i] - feh[i]
preal = np.array([alphafe, alpha[i], alpha[i], alpha[i], alpha[i], alpha[i], alpha[i],
                  feh[i], Teff[i], logg[i], dv[i]])

spec_ind = matches[i]
full_spec = full_spectra[spec_ind]
norm_spec = norm_spectra[spec_ind]
spec_err = spectral_err[spec_ind]

p0 = [0, 0, 0, 0, 0, 0, 0, 0, 5000, 4, 0]
popt, pcov, model_spec \
    = fitting.fit_normalized_spectrum_single_star_model(norm_spec = norm_spec,
                                                        spec_err = spec_err,
                                                        NN_coeffs = NN_coeffs,
                                                        p0 = p0, num_p0 = 10)

# Define Likelihood function and priors
def lnlike(labels, data_spec, data_err):
    model_spec = NN.get_spectrum_from_neural_net(labels=labels, NN_coeffs=NN_coeffs)
    inv_sigma2 = 1.0/data_err
    lnchi2 = -0.5 * (np.sum((data_spec - model_spec)**2 * inv_sigma2))
    return(lnchi2)


def lnprior(labels):
    abundances = labels[:-3]
    Teff = labels[-3]
    logg = labels[-2]
    dv = labels[-1]
    if np.any(abundances < -10) or np.any(abundances > 10.0):
        return(-np.inf)
    elif (Teff < 3000 or Teff > 10000):
        return(-np.inf)
    elif (logg < 0.01 or logg > 20):
        return(-np.inf)
    elif (dv < -50 or dv > 50):
        return(-np.inf)
    else:
        return(1)


def lnprob(labels, data_spec, data_err):
    lp = lnprior(labels)
    if not np.isfinite(lp):
        return(-np.inf)
    else:
        return(lp + lnlike(labels, data_spec, data_err))


# Initialize MCMC
ndim = len(popt)
nwalkers = 128

filename = '/global/scratch/nathan_sandford/emcee/chain.h5'
backend = emcee.backends.HDFBackend(filename)

try: # If chain has already been run for a while
    previous_autocorr = np.load('/global/scratch/nathan_sandford/emcee/autocorr.npy')
    extension = np.zeros(nsteps // 100)
    autocorr = np.concatenate((previous_autocorr, extension))
    previous_steps = np.len(backend.get_chain()[0])
    p0 = backend.get_last_sample()[0]
    old_tau = backend.get_autocorr_time(tol=0)
    print('Loaded chain with %i steps run' % previous_steps)
except FileNotFoundError: # If chain is being run for the first time
    autocorr = np.empty(nsteps // 100)
    previous_steps = 0
    backend.reset(nwalkers, ndim)
    p0 = popt + 1e-2*np.random.uniform(low=-1.0, high=1.0, size=(nwalkers, ndim))  # Initialize at best fit from above
    old_tau = np.inf
    print('Initialized new chain')
index = previous_steps
    


with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(norm_spec, spec_err), backend=backend, pool=pool)

    for sample in sampler.sample(p0, iterations=nsteps, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print('Converged!')
            break
        old_tau = tau

np.save('/global/scratch/nathan_sandford/emcee/autocorr.npy', autocorr)
os.system('cp /global/scratch/nathan_sandford/emcee/autocorr.npy '
          + '/global/scratch/nathan_sandford/emcee/autocorr_%i.npy' % (previous_steps + nsteps))

print('Now completed %i iterations' % (previous_steps + nsteps))
