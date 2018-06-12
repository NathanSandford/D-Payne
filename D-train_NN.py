'''
Script to train NN's on theoretical spectra from the Kurucz/ATLAS12 Models

Adapted from Kareem's
binspec/train_NN/train_NNs/train_NN_spectral_model.py and
Yuan-Sen's train-NN.py
'''

# Import Packages
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import multiprocessing
from multiprocessing import Pool

'''
Print Updates?
'''
verbose = False

'''
Multiprocessing and Batching Parameters
'''
num_CPU = multiprocessing.cpu_count()  # Number of cores available
pixel_batch_size = 250  # Number of pixels to train at a time
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)  # Number of threads per CPU
if verbose:
    print('Number of CPU: %i' % num_CPU)
    print('Pixel batch size: %i' % pixel_batch_size)


'''
Select Training Batches
'''
num_start = int(sys.argv[1])
num_end = int(sys.argv[2])


'''
Restore Training Spectra
'''
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'
InputDir = D_PayneDir + 'spectra/synth_spectra/'
TrainingSpectraFile = 'convolved_synthetic_spectra_MIST.npz'
TrainingSpectra = np.load(InputDir+TrainingSpectraFile)
n_spectra = len(TrainingSpectra['labels'])


'''
Train on Normalized Spectra?
If so, train on theoretically normalized spectra?
'''
normalized = True
perfect_normalization = False


'''
Choose Labels
See combine_spectra.py for list of labels saved from Kurucz/Atlas12 models
'''
label_ind = [0, -3, -2, -1]  # [alpha/Fe], [Fe/H], Teff, logg
labels = TrainingSpectra['labels'][:, label_ind]


'''
Size of training set.
Anything over a few 100 should be OK for a small network (see YST's paper),
but it can't hurt to go larger if the training set is available.
'''
n_train = np.int(np.floor(n_spectra * 4/5))  # Size of training set
n_valid = np.int(n_spectra - n_train)  # Size of validating set


'''
Randomly Select Training and Validating Set
'''
train_ind = sorted(np.random.choice(n_spectra, n_train, replace=False))
valid_ind = np.delete(np.arange(n_spectra), train_ind)


for num_go in range(num_start, num_end + 1):
    if verbose:
        print('==============================================================')
        print('Starting batch %d/%d' % (num_go+1, num_end+1))

    '''
    Restore Training Set
    '''
    if normalized is False:  # Restore unnormed spectra
        if verbose:
            print('Restoring unnormalized synthetic spectra...')
        spec_key = 'spectra'
    elif perfect_normalization is True:  # Restore theoretically normed spectra
        if verbose:
            print('Restoring perfectly normalized synthetic spectra...')
        spec_key = 'norm_spectra_true'
    elif perfect_normalization is False:  # Restore approx. normed spectra
        if verbose:
            print('Restoring imperfectly normalized synthetic spectra...')
        spec_key = 'norm_spectra_approx'
    spec = TrainingSpectra[spec_key]
    x = labels[train_ind, :]
    y = spec[train_ind,
             num_go * pixel_batch_size:(num_go+1) *
             pixel_batch_size]

    '''
    Validating Set
    '''
    x_valid = labels[valid_ind, :]
    y_valid = spec[valid_ind,
                   num_go * pixel_batch_size:(num_go+1) *
                   pixel_batch_size]

    '''
    Scale the Labels
    '''
    if verbose:
        print('Scaling labels...')
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min) - 0.5
    x_valid = (x_valid-x_min)/(x_max-x_min) - 0.5

    '''
    Bail if you've reached the last pixel in the spectrum--
    Not sure if this is working successfully
    '''
    if not len(y):
        if verbose:
            print('Reached the end of the spectrum!')
        continue

    '''
    Dimension of the Input
    '''
    dim_in = x.shape[1]
    num_pix = y.shape[1]

    '''
    Make pytorch Variables
    '''
    if verbose:
        print('Making PyTorch variables (x)...')
    x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
    if verbose:
        print('Making PyTorch variables (y)...')
    y = Variable(torch.from_numpy(y),
                 requires_grad=False).type(torch.FloatTensor)
    if verbose:
        print('Making PyTorch variables (x_valid)...')
    x_valid = Variable(torch.from_numpy(x_valid)).type(torch.FloatTensor)
    if verbose:
        print('Making PyTorch variables (y_valid)...')
    y_valid = Variable(torch.from_numpy(y_valid),
                       requires_grad=False).type(torch.FloatTensor)
    if verbose:
        print('Made all PyTorch variables!')

    '''
    Loop over all pixels
    '''
    def train_pixel(pixel_no):
        '''
        Trains a NN for a single pixel.
        To be fed to multiprocessing.
        '''

        # Define neural network
        print('Defining neural network...')
        model = torch.nn.Sequential(torch.nn.Linear(dim_in, 10),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(10, 1),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(1, 1)
                                    )

        # Define Optimizer
        learning_rate = 0.001
        print('Defining optimizer w/ learning rate: %.3f...' % learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Convergence Counter
        current_loss = np.inf
        count = 0
        t = 0

        # Train the Neural Network
        if verbose:
            print('Training neural network on pixel %i' % pixel_no)
        while count < 5:

            # Training
            y_pred = model(x)[:, 0]
            loss = ((y_pred-y[:, pixel_no]).pow(2)/(0.01**2)).mean()
            # Validation
            y_pred_valid = model(x_valid)[:, 0]
            loss_valid = (((y_pred_valid-y_valid[:, pixel_no]).pow(2) /
                           (0.01**2)).mean()).data[0]

            # Check Convergence
            if t % 10000 == 0:
                if verbose:
                    print('Checking convergence... (Pixel #: %i/ Count = %i)'
                          % (pixel_no, count))
                if loss_valid > current_loss:
                    count += 1
                else:
                    # Record the best loss
                    current_loss = loss_valid

                    # Record the best parameters
                    model_numpy = []
                    for param in model.parameters():
                        model_numpy.append(param.data.numpy())

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1

        # Return Parameters
        if verbose:
            print('Completed training on pixel %i, batch %i'
                  % (pixel_no, num_go+1))
        return(model_numpy)

    '''
    Train in Parallel
    '''
    if verbose:
        print('Initializing pool...')
    pool = Pool(num_CPU)
    if verbose:
        print('Beginning training in parallel!')
    net_array = pool.map(train_pixel, range(num_pix))
    if verbose:
        print('Trained all pixels for batch %i!' % (num_go+1))

    '''
    Extract Parameters
    '''
    if verbose:
        print('Extracting parameters for batch %i...' % (num_go+1))
    w_array_0 = np.array([net_array[i][0] for i in range(len(net_array))])
    b_array_0 = np.array([net_array[i][1] for i in range(len(net_array))])
    w_array_1 = np.array([net_array[i][2][0] for i in range(len(net_array))])
    b_array_1 = np.array([net_array[i][3][0] for i in range(len(net_array))])
    w_array_2 = np.array([net_array[i][4][0][0] for i in range(len(net_array))])
    b_array_2 = np.array([net_array[i][5][0] for i in range(len(net_array))])

    '''
    Save parameters and remember how we scale the labels
    '''
    if verbose:
        print('Saving parameters for batch %i' % (num_go+1))
    num_go_str = '0'+str(num_go) if num_go < 10 else str(num_go)
    np.savez(D_PayneDir + "neural_nets/NN_results_" + num_go_str + ".npz",
             w_array_0=w_array_0, w_array_1=w_array_1, w_array_2=w_array_2,
             b_array_0=b_array_0, b_array_1=b_array_1, b_array_2=b_array_2,
             x_max=x_max, x_min=x_min)
