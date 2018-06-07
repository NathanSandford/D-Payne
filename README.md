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

# References
The Payne is spectral fitting technique developed in [Yuan-Sen Ting et al. in prep](https://arxiv.org/abs/1804.01530).
</br> </br>
Additionally, much of the code here as been adapted from Kareem El-Badry's adaptation of the Payne for fitting binary spectra
([El-Badry et. al. 2018a](http://adsabs.harvard.edu/doi/10.1093/mnras/sty240) 
and [El-Badry et al. 2018b](http://adsabs.harvard.edu/abs/2018MNRAS.473.5043E)) which can be found
[here](https://github.com/kareemelbadry/binspec) package.
