# import package
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
from multiprocessing import Pool

# set number of threads per CPU
import mkl
mkl.set_num_threads(1)

#------------------------------------------------------------------------------
# number of CPUs for parallel computing
num_CPU = 16

# choose a testing batch
num_go = int(sys.argv[1])


#==============================================================================
# restore training spectra
temp = np.load("/data/ting/mock_all_spectra_no_noise.npz")
x = (temp["labels"].T)[:1000,:]
y = temp["spectra"][:1000,num_go*150:(num_go+1)*150]

# and validation spectra
x_valid = (temp["labels"].T)[1000:1200,:]
y_valid = temp["spectra"][1000:1200,num_go*150:(num_go+1)*150]

# scale the labels
x_max = np.max(x, axis=0)
x_min = np.min(x, axis=0)
x = (x-x_min)/(x_max-x_min) - 0.5
x_valid = (x_valid-x_min)/(x_max-x_min) - 0.5

#-----------------------------------------------------------------------------
# dimension of the input
dim_in = x.shape[1]
num_pix = y.shape[1]

# make pytorch variables
x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
y = Variable(torch.from_numpy(y), requires_grad=False).type(torch.FloatTensor)
x_valid = Variable(torch.from_numpy(x_valid)).type(torch.FloatTensor)
y_valid = Variable(torch.from_numpy(y_valid),\
                   requires_grad=False).type(torch.FloatTensor)


#=============================================================================
# loop over all pixels
def train_pixel(pixel_no):
    
    # define neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(1,1)
    )

    # define optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#-----------------------------------------------------------------------------
    # convergence counter
    current_loss = np.inf
    count = 0
    t = 0

#-----------------------------------------------------------------------------
    # train the neural network
    while count < 5:

        # training
        y_pred = model(x)[:,0]
        loss = ((y_pred-y[:,pixel_no]).pow(2)/(0.01**2)).mean()

        # validation
        y_pred_valid = model(x_valid)[:,0]
        loss_valid = (((y_pred_valid-y_valid[:,pixel_no]).pow(2)\
                       /(0.01**2)).mean()).data[0]

        
#==============================================================================
        # check convergence
        if t % 10000 == 0:
            if loss_valid > current_loss:
                count += 1
            else:
                # record the best loss
                current_loss = loss_valid

                # record the best parameters
                model_numpy = []
                for param in model.parameters():
                    model_numpy.append(param.data.numpy())
                
#-----------------------------------------------------------------------------
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1
         
#-----------------------------------------------------------------------------
    # return parameters
    return model_numpy    


#=============================================================================
# train in parallel
pool = Pool(num_CPU)
net_array = pool.map(train_pixel,range(num_pix))

# extract parameters
w_array_0 = np.array([net_array[i][0] for i in range(len(net_array))])
b_array_0 = np.array([net_array[i][1] for i in range(len(net_array))])
w_array_1 = np.array([net_array[i][2][0] for i in range(len(net_array))])
b_array_1 = np.array([net_array[i][3][0] for i in range(len(net_array))])
w_array_2 = np.array([net_array[i][4][0][0] for i in range(len(net_array))])
b_array_2 = np.array([net_array[i][5][0] for i in range(len(net_array))])

# save parameters and remember how we scale the labels
np.savez("/data/ting/NN_results_" \
         + str(num_go) + ".npz",\
         w_array_0 = w_array_0,\
         w_array_1 = w_array_1,\
         w_array_2 = w_array_2,\
         b_array_0 = b_array_0,\
         b_array_1 = b_array_1,\
         b_array_2 = b_array_2,\
         x_max=x_max,\
         x_min=x_min)
