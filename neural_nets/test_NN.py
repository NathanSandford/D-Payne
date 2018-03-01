# import packages
import numpy as np
import sys
import os
from multiprocessing import Pool
from scipy import interpolate
from scipy.optimize import curve_fit

# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)


#=====================================================================
# number of processor
num_CPU = 16

# number of labels
num_labels = 26

# testing batch
num_go = int(sys.argv[1])


#======================================================================
# restore testing spectra
temp  = np.load("apogee_spectra_"+str(num_go)+".npz")

Y_u_all = temp["spectra"].T
Y_u_all_err = temp["spectra_err"].T


#======================================================================
# load NN results
temp = np.load("NN_results_train_1000.npz")
w_array_0 = temp["w_array_0"]
w_array_1 = temp["w_array_1"]
w_array_2 = temp["w_array_2"]
b_array_0 = temp["b_array_0"]
b_array_1 = temp["b_array_1"]
b_array_2 = temp["b_array_2"]
x_min = temp["x_min"]
x_max = temp["x_max"]

#-----------------------------------------------------------------------
# define sigmoid function
def sigmoid_def(z):
    return 1.0/(1.0+np.exp(-z))


#=======================================================================
# restore wavelength grid
temp = np.load("apogee_windows.npz")
wavelength_template = temp["filter_wavelength"]

# mask considering line list residual
temp = np.load("arcturus_apogee.npz")
flux_apogee = temp["convolved_flux"]
temp = np.load("arcturus_cargile.npz")
flux_cargile = temp["convolved_flux"]

chowin = np.abs(flux_apogee-flux_cargile) > 0.02
Y_u_all_err[chowin,:] = 999.

temp = np.load("solar_apogee.npz")
flux_apogee = temp["convolved_flux"]
temp = np.load("solar_cargile.npz")
flux_cargile = temp["convolved_flux"]

chowin = np.abs(flux_apogee-flux_cargile) > 0.02
Y_u_all_err[chowin,:] = 999.


#===========================================================================
# fit best models
def fit_func(input_param, *labels):
    predict_flux = w_array_2*sigmoid_def(np.sum(w_array_1*(sigmoid_def(np.dot(\
                    w_array_0,labels[:-1])+b_array_0)),axis=1)+b_array_1)\
                    +b_array_2
    f_interp = interpolate.interp1d(wavelength_template, predict_flux,\
                bounds_error=False, kind="linear", fill_value="extrapolate")
    return f_interp(wavelength_template+labels[-1]*wavelength_template/10**5)


#---------------------------------------------------------------------------
# define function to perform testing step in batch
def fit_spectrum(spec_no):
    p0_test = np.zeros(num_labels)
    
    try:
        popt, pcov = curve_fit(fit_func, [spec_no], Y_u_all[:,spec_no],\
                                   p0 = p0_test,\
                                   sigma=Y_u_all_err[:,spec_no],\
                                   absolute_sigma=True, bounds=(-0.5,0.5))
    except:
        popt = np.zeros(num_labels) - 9999.

    return popt


#============================================================================
# fit spectra
pool = Pool(num_CPU)
recovered_results = np.array(pool.map(fit_spectrum,range(Y_u_all.shape[1]))).T

#-------------------------------------------------------------------------------
# initiate chi^2
chi2 = []

# loop over all spectra
for j in range(recovered_results.shape[1]):
    predict_flux = w_array_2*sigmoid_def(np.sum(w_array_1*(sigmoid_def(np.dot(\
            w_array_0,recovered_results[:-1,j])+b_array_0)),axis=1)+b_array_1)\
            +b_array_2
    f_interp = interpolate.interp1d(wavelength_template, predict_flux,\
                bounds_error=False, kind="linear", fill_value="extrapolate")
    predict_flux = f_interp(wavelength_template \
                            + recovered_results[-1,j]*wavelength_template/10**5)
    chi2.append(np.mean((predict_flux-Y_u_all[:,j])**2/(Y_u_all_err[:,j]**2)))
chi2 = np.array(chi2)


#----------------------------------------------------------------------------
# rescale back to original values
for i in range(recovered_results.shape[0]-1):
    ind_invalid = (recovered_results[i,:] < -100.)
    recovered_results[i,:] = (recovered_results[i,:]+0.5)*(x_max[i]-x_min[i])\
                             +x_min[i]
    recovered_results[i,ind_invalid] = -999.

# save array
np.savez("mock_all_predictions_" + str(num_go) + ".npz",\
         abundance_prediction=recovered_results[:-1,:], chi2=chi2,\
         radial_shift=recovered_results[-1,:],\
         num_pix = np.sum(Y_u_all_err[:,0] != 999.))
