import numpy as np
import glob
import os

# Laptop D-Payne Directroy
D_PayneDir = \
    '/Users/Nathan/Documents/Berkeley/Chemical_Evolution/DEIMOS/D-Payne/'

# Savio D-Payne Directroy
# D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'

# Neural Network Directory
NN_Dir = D_PayneDir + 'neural_nets/'

# Name of combined results
NN_out = 'NN_norm_spectra_approx'

# Find all NN_results files
NN_list = [os.path.basename(x) for x in glob.glob(NN_Dir+'*') if
           os.path.basename(x)[11].isdigit()]
NN_list.sort()

# Initialize lists for all NN coefficients
w_array_0 = []
w_array_1 = []
w_array_2 = []
b_array_0 = []
b_array_1 = []
b_array_2 = []
x_min = []
x_max = []

# Restore NN coefficients for all NN_results files
for file in NN_list:
    temp = np.load(NN_Dir+file)
    try:  # Try to load NN w/ 2 hidden layers
        w_array_0.append(temp['w_array_0'])
        w_array_1.append(temp['w_array_1'])
        w_array_2.append(temp['w_array_2'])
        b_array_0.append(temp['b_array_0'])
        b_array_1.append(temp['b_array_1'])
        b_array_2.append(temp['b_array_2'])
        x_min.append(temp['x_min'])
        x_max.append(temp['x_max'])
        n_hidden = 2
    except KeyError:  # Load NN w/ only 1 hidden layers
        w_array_0.append(temp['w_array_0'])
        w_array_1.append(temp['w_array_1'])
        b_array_0.append(temp['b_array_0'])
        b_array_1.append(temp['b_array_1'])
        x_min.append(temp['x_min'])
        x_max.append(temp['x_max'])
        n_hidden = 1
    temp.close()

# Concatenate all NN coefficients
if n_hidden == 2:
    w_array_0 = np.concatenate(w_array_0, axis=0)
    w_array_1 = np.concatenate(w_array_1, axis=0)
    w_array_2 = np.concatenate(w_array_2, axis=0)
    b_array_0 = np.concatenate(b_array_0, axis=0)
    b_array_1 = np.concatenate(b_array_1, axis=0)
    b_array_2 = np.concatenate(b_array_2, axis=0)
elif n_hidden == 1:
    w_array_0 = np.concatenate(w_array_0, axis=0)
    w_array_1 = np.concatenate(w_array_1, axis=0)
    b_array_0 = np.concatenate(b_array_0, axis=0)
    b_array_1 = np.concatenate(b_array_1, axis=0)

'''
x_min and x_max are ALMOST the same for all NN_results files
so I'm just taking the first set of x_min and x_max... That may be a mistake...
We'll see!
'''
x_min = x_min[0]
x_max = x_max[0]

# Save combined NN coefficients
if n_hidden == 2:
    np.savez(NN_Dir+NN_out+'.npz',
             w_array_0=w_array_0, w_array_1=w_array_1, w_array_2=w_array_2,
             b_array_0=b_array_0, b_array_1=b_array_1, b_array_2=b_array_2,
             x_max=x_max, x_min=x_min)

elif n_hidden == 1:
    np.savez(NN_Dir+NN_out+'.npz',
             w_array_0=w_array_0, w_array_1=w_array_1,
             b_array_0=b_array_0, b_array_1=b_array_1,
             x_max=x_max, x_min=x_min)

# Clean up
os.system('mkdir '+NN_out)
os.system('mv NN_results* '+NN_out)
os.system('cp '+NN_out+'.npz '+NN_out)
os.system('zip '+NN_out+'.zip '+NN_out)
os.system('rmdir -r '+NN_out)
