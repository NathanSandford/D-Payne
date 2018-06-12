# D-Payne
Fitting DEIMOS spectra with the Payne

# Table of Contents
* utils.py: contains a bunch of frequently used functions
* convolve_highrez_spectra_to_given_R.py: Script to batch convolve high-resolution synthetic spectra down to the resolution of your observed spectra
* normalize_synth_spectra.py: Script to batch normalize synthetic spectra using a gaussian smoothing of continuum regions
* process_spectra.py: script to further processes spectra—normalization and RV Correction—Actually out of date, see development/process_spectra_batch.ipynb
* model_spectra.py: Functions for generating model spectra from NN
* fitting.py: Functions for fitting abundances with scipy.curve_fit() optimizer

* train_NNs/D-train_NN.py: script to train NN on synthetic spectra
* neural_nets/combine_NN.py: script to combine all NN outputs quickly

* other_data/create_deimos_wavelength_template.py: Script to create wavelength template for DEIMOS, deimos_wavelength.npz
* other_data/determine_continuum_regions.py: indices of wavelength template to consider continuum regions, deimos_cont_pixels.npy
* other_data/standards/ includes lists of stars with measured abundances to create masks (e.g. mask.008.0010337.npy)

* jobs/ contains SLURM job scripts for the more computationally intensive scripts

* development/ contains a lot of Jupyter Notebooks that I've been using to develop and test new code

# Pipeline Flow
0.1 Reduce Observed data w/ DEEP2 Pipeline <br>
0.2 Generate a Training Set of Synthetic Spectra (See NathanSandford/kurucz) <br>

1. Pre-Process Observed Spectra: development/process_spectra_batch.ipynb <br>
  1.1 Apply Telluric Masks <br>
  1.2 RV shift <br>
  1.3 Normalize <br>
  
2. Train Neural Network <br>
  2.1 Convolve Synthetic Spectra to DEIMOS resolution: convolve_highrez_spectra_to_given_R.job <br>
  2.2 Normalize Synthetic Spectra like DEIMOS: normalize_synth_spectra.job <br>
  2.3 Train: D-train_NN_\*.job <br>
  
3. Generate Masks (Optional) <br>
  3.1 Find Overlapping Sample: development/Standard_Stars.ipynb <br>
  3.2 Generate Masks: development/Masking.ipynb <br>

4. Fit Labels <br>
  4.a Fit w/ optimizer: /development/Testing_NN_fitting.ipynb <br>
  4.b Fit w/ MCMC: /development/emcee/


# References
The Payne is spectral fitting technique developed in [Yuan-Sen Ting et al. in prep](https://arxiv.org/abs/1804.01530).
</br> </br>
Additionally, much of the code here as been adapted from Kareem El-Badry's adaptation of the Payne for fitting binary spectra
([El-Badry et. al. 2018a](http://adsabs.harvard.edu/doi/10.1093/mnras/sty240) 
and [El-Badry et al. 2018b](http://adsabs.harvard.edu/abs/2018MNRAS.473.5043E)) which can be found
[here](https://github.com/kareemelbadry/binspec).
