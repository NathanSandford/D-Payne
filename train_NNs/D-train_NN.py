'''
Adapted from Kareem's
binspec/train_NN/train_NNs/train_NN_spectral_model.py and
Yuan-Sen's train-NN.py
'''

# import package
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import multiprocessing
from multiprocessing import Pool

# ------------------------------------------------------------------------------
num_CPU = multiprocessing.cpu_count()
print('Number of CPU: %i' % num_CPU)
pixel_batch_size = 250
print('Pixel batch size: %i' % pixel_batch_size)

# set number of threads per CPU
os.environ['OMP_NUM_THREADS'] = '{:d}'.format(1)

# choose a testing batch
num_start = int(sys.argv[1])
num_end = int(sys.argv[2])

# restore training spectra
D_PayneDir = '/global/home/users/nathan_sandford/D-Payne/'
InputDir = D_PayneDir + 'spectra/synth_spectra/'
TrainingSpectraFile = 'convolved_synthetic_spectra_MIST.npz'
TrainingSpectra = np.load(InputDir+TrainingSpectraFile)
n_spectra = len(TrainingSpectra['labels'])
normalized = True
perfect_normalization = False

'''
Choose Labels.
See combine_spectra.py for list of labels saved from Kurucz/Atlas12 models
'''
label_ind = [0, -3, -2, -1]  # [alpha/Fe], [Fe/H], Teff, logg
labels = TrainingSpectra['labels'][:, label_ind]

'''
Size of training set.
Anything over a few 100 should be OK for a small network (see YST's paper),
but it can't hurt to go larger if the training set is available.
'''
n_train = np.int(np.floor(n_spectra * 4/5))
n_valid = np.int(n_spectra - n_train)
train_ind = sorted(np.random.choice(n_spectra, n_train, replace=False))
valid_ind = np.delete(np.arange(n_spectra), train_ind)

for num_go in range(num_start, num_end + 1):
    print('==================================================================')
    print('Starting batch %d/%d' % (num_go+1, num_end+1))
    # =========================================================================
    # restore training spectra
    if normalized is False:
        print('Restoring unnormalized synthetic spectra...')
        spec_key = 'spectra'
    elif perfect_normalization is True:
        print('Restoring perfectly normalized synthetic spectra...')
        spec_key = 'norm_spectra_true'
    elif perfect_normalization is False:
        print('Restoring imperfectly normalized synthetic spectra...')
        spec_key = 'norm_spectra_approx'
    spec = TrainingSpectra[spec_key]

    x = labels[train_ind, :]
    y = spec[train_ind,
             num_go * pixel_batch_size:(num_go+1) *
             pixel_batch_size]

    # and validation spectra
    x_valid = labels[valid_ind, :]
    y_valid = spec[valid_ind,
                   num_go * pixel_batch_size:(num_go+1) *
                   pixel_batch_size]

    # scale the labels
    print('Scaling labels...')
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x = (x-x_min)/(x_max-x_min) - 0.5
    x_valid = (x_valid-x_min)/(x_max-x_min) - 0.5

    # if you've reached the last pixel in the spectrum
    if not len(y):
        print('Reached the end of the spectrum!')
        continue

    # -------------------------------------------------------------------------
    # dimension of the input
    dim_in = x.shape[1]
    num_pix = y.shape[1]

    # make pytorch variables
    print('Making PyTorch variables (x)...')
    x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
    print('Making PyTorch variables (y)...')
    y = Variable(torch.from_numpy(y),
                 requires_grad=False).type(torch.FloatTensor)
    print('Making PyTorch variables (x_valid)...')
    x_valid = Variable(torch.from_numpy(x_valid)).type(torch.FloatTensor)
    print('Making PyTorch variables (y_valid)...')
    y_valid = Variable(torch.from_numpy(y_valid),
                       requires_grad=False).type(torch.FloatTensor)
    print('Made all PyTorch variables!')

    # =======================================================================
    # loop over all pixels
    def train_pixel(pixel_no):
        '''
        to be fed to multiprocessing
        '''

        # define neural network
        print('Defining neural network...')
        model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1, 1)
        )

        # define optimizer
        learning_rate = 0.001
        print('Defining optimizer w/ learning rate: %.3f...' % learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # ---------------------------------------------------------------------------
        # convergence counter
        current_loss = np.inf
        count = 0
        t = 0

        # ---------------------------------------------------------------------------
        # train the neural network
        print('Training neural network on pixel %i' % pixel_no)
        while count < 5:

            # training
            y_pred = model(x)[:, 0]
            loss = ((y_pred-y[:, pixel_no]).pow(2)/(0.01**2)).mean()

            # validation
            y_pred_valid = model(x_valid)[:, 0]
            loss_valid = (((y_pred_valid-y_valid[:, pixel_no]).pow(2) /
                           (0.01**2)).mean()).data[0]

            # =======================================================================
            # check convergence
            if t % 10000 == 0:
                print('Checking convergence... (Pixel #: %i/ Count = %i)' %
                      (pixel_no, count))
                if loss_valid > current_loss:
                    count += 1
                else:
                    # record the best loss
                    current_loss = loss_valid

                    # record the best parameters
                    model_numpy = []
                    for param in model.parameters():
                        model_numpy.append(param.data.numpy())

            # -----------------------------------------------------------------------
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1

        # ---------------------------------------------------------------------------
        # return parameters
        print('Completed training on pixel %i, batch %i' %
              (pixel_no, num_go+1))
        return model_numpy

    # ===============================================================================
    # train in parallel
    print('Initializing pool...')
    pool = Pool(num_CPU)
    print('Beginning training in parallel!')
    net_array = pool.map(train_pixel, range(num_pix))
    print('Trained all pixels for batch %i!' % (num_go+1))

    # extract parameters
    print('Extracting parameters for batch %i...' % (num_go+1))
    w_array_0 = np.array([net_array[i][0] for i in range(len(net_array))])
    b_array_0 = np.array([net_array[i][1] for i in range(len(net_array))])
    w_array_1 = np.array([net_array[i][2][0] for i in range(len(net_array))])
    b_array_1 = np.array([net_array[i][3][0] for i in range(len(net_array))])
    w_array_2 = np.array([net_array[i][4][0][0] for i in range(len(net_array))])
    b_array_2 = np.array([net_array[i][5][0] for i in range(len(net_array))])

    # save parameters and remember how we scale the labels
    print('Saving parameters for batch %i' % (num_go+1))
    num_go_str = '0'+str(num_go) if num_go < 10 else str(num_go)
    np.savez(D_PayneDir + "neural_nets/NN_results_" + num_go_str + ".npz",
             w_array_0=w_array_0, w_array_1=w_array_1, w_array_2=w_array_2,
             b_array_0=b_array_0, b_array_1=b_array_1, b_array_2=b_array_2,
             x_max=x_max, x_min=x_min)
