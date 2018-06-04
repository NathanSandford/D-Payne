import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import emcee
from emcee import PTSampler

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
p0 = popt + 1e-2*np.random.uniform(low=-1.0, high=1.0, size=(nwalkers, ndim))  # Initialize at best fit from above

# Run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(norm_spec, spec_err))
pos, prob, state = sampler.run_mcmc(p0,500000)

# Plot MCMC Results
for i in range(ndim):
    plt.plot(sampler.chain[:,:,i].T, '-', alpha=0.1)
    plt.savefig('chains_dim%i.png' %i)

chain = sampler.get_chain()[:, :, 0].T
fig = corner.corner(chain)
plt.savefig('cornerplot.png')



# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]


# Compute the estimators for a few different chain lengths
N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
gw2010 = np.empty(len(N))
new = np.empty(len(N))
for i, n in enumerate(N):
    gw2010[i] = autocorr_gw2010(chain[:, :n])
    new[i] = autocorr_new(chain[:, :n])

# Plot the comparisons
plt.loglog(N, gw2010, "o-", label="G\&W 2010")
plt.loglog(N, new, "o-", label="new")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14)
plt.savefig('Auto_Correlation.png')
