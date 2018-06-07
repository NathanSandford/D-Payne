# Python2 compatibility
from __future__ import absolute_import, division, print_function
import numpy as np
import multiprocessing
from multiprocessing import Pool
import fitting
import utils

import sys
import os

from scipy import interpolate
from scipy.optimize import curve_fit


#=====================================================================
# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)

# number of processors
num_CPU = multiprocessing.cpu_count()

# number of labels
num_labels = 26

# testing batch
num_go = int(sys.argv[1])

# Restore Neural Network
NN_coeffs = utils.read_in_neural_network(name='norm_spectra_approx')









