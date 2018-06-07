import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee
import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import corner

import utils
import model_spectra as NN
import fitting

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
nstep = 50000
num_CPU = multiprocessing.cpu_count()
p0 = popt + 1e-2*np.random.uniform(low=-1.0, high=1.0, size=(nwalkers, ndim))  # Initialize at best fit from above
index = 0
autocorr = np.empty(nstep)
old_tau = np.inf

backend = emcee.backends.HDFBackend('/global/scratch/nathan_sandford/emcee/chain.h5')
backend.reset(nwalkers, ndim)

# Run MCMC
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(norm_spec, spec_err),
                                    backend=backend, pool=pool)

    for sample in sampler.sample(p0, iterations=nstep, progress=True):
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
            break
        old_tau = tau



# Plot Autocorrelation
n = 100*np.arange(1, index+1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");
plt.savefig('/global/scratch/nathan_sandford/emcee/AutoCorr_50000.png')



# Plot corner
tau = sampler.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate((
    samples, log_prob_samples[:, None], log_prior_samples[:, None]
), axis=1)

labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
labels += ["log prob", "log prior"]

fig = corner.corner(all_samples, labels=labels)
fig.savefig('/global/scratch/nathan_sandford/emcee/Corner_50000.png')
