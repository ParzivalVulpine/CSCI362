#!/usr/bin/env python3
# nonlinear_7.py                                                      SSimmons Oct. 2018
"""
Uses the nn library along with autograd to find the least-squares linear model
using, optionally, mini-batch gradient descent along and/or momentum.

The data are mean centered and normalized by default.  The plane obtained by
gradient descent is compared with the one obtained by solving the normal
equations linear algebraically.
"""
import argparse
import csv

import torch
import torch.nn as nn

# Read in the data.
with open('temp_co2_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # skip the first line of csvfile
    xss, yss = [], []
    for row in reader:
        xss.append([float(row[2]), float(row[3])])
        yss.append([float(row[1])])
num_examples = len(xss)  # the number of examples

# Sort out the command line options.
parser = argparse.ArgumentParser(
    description= \
        "Mini-batch gradient descent with momentum training a linear model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
p = parser.add_argument
p("-nc", "--nocenter", help="do not center", action="store_true")
p("-nn", "--nonorm", help="do not normalize", action="store_true")
p("-e", "--epochs", help="no. of epochs", type=int, default=400)
p("-lr", "--alpha", help="learning rate", type=float, default=0.032)
p("-mo", "--beta", help="momentum", type=float, default=0.0)
help_str = "batch size; default is batch gradient descent"
p("-bs", "--batchsize", help=help_str, type=int, default=num_examples)
args = parser.parse_args()

# The tensors xss and yss now contain the features and targets of the data.
xss, yss = torch.tensor(xss), torch.tensor(yss)

# For final validation of our trained model, we compute here the least-squares
# regression plane using (built-in) numerically stable linear alg. algorithms.
A = torch.cat((torch.ones(len(xss), 1), xss), 1)
lin_alg_sol = torch.linalg.lstsq(A, yss, driver='gels').solution[:, 0]

# Compute the column-wise means and standard deviations.
if not args.nocenter:
    xss_means, yss_means = xss.mean(0), yss.mean()
if not args.nonorm:
    xss_stds, yss_stds = xss.std(0), yss.std()

# Mean-center and normalize (if the means and stdevs were computed above).
if 'xss_means' in locals() and 'yss_means' in locals():
    xss.sub_(xss_means), yss.sub_(yss_means)
if 'xss_stds' in locals() and 'yss_stds' in locals():
    xss.div_(xss_stds), yss.div_(yss_stds)


# Build the model class
class LinearRegressionModel(nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, xss):
        return self.layer(xss)


# Create and print an instance of the above class
model = LinearRegressionModel()
print(model)

# This is used to implement momentum (by hand) -- specifically, it remembers the
# values of the weights from the previous iteration during the training loop.
z_parameters = []
for param in model.parameters():
    z_parameters.append(param.data.clone())
for param in z_parameters:
    param.zero_()

# Set the criterion.
criterion = nn.MSELoss()

# Train the model.
for epoch in range(args.epochs):

    accum_loss = 0
    indices = torch.randperm(32)

    for idx in range(0, num_examples, args.batchsize):

        # forward pass
        current_indices = indices[idx:idx + args.batchsize]
        yss_pred = model(xss[current_indices])
        loss = criterion(yss_pred, yss[current_indices])

        accum_loss += loss.item()

        # back propagation
        model.zero_grad()
        loss.backward()

        # adjust the weights
        for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
            z_parameters[i] = args.beta * z_param + param.grad.data
            param.data.sub_(z_parameters[i] * args.alpha)

    print_str = "epoch: {0}, loss: {1}". \
        format(epoch + 1, accum_loss * args.batchsize / num_examples)
    if args.epochs < 20 or epoch < 7 or epoch > args.epochs - 17:
        print(print_str)
    elif epoch == 7:
        print("...")
    else:
        print(print_str, end='\b' * len(print_str), flush=True)

# Extract the weights and bias into a list.
params = list(model.parameters())

# Un-mean-center and un-normalize the weights (for final validation against
# the OLS regression plane found above using linear algebra).
if not ('xss_means' in locals() and 'yss_means' in locals()):
    xss_means = torch.Tensor([0., 0.]);
    yss_means = 0.0
    print('no centering, ', end='')
else:
    print('centered, ', end='')
if not ('xss_stds' in locals() and 'yss_stds' in locals()):
    xss_stds = torch.Tensor([1., 1.]);
    yss_stds = 1.0;
    print('no normalization, ', end='')
else:
    print('normalized, ', end='')
w = torch.zeros(3)
w[1:] = params[0] * yss_stds / xss_stds
w[0] = params[1].data.item() * yss_stds + yss_means - w[1:] @ xss_means

# Print the hyper-paramters used to train the model.
print("learning rate: " + str(args.alpha) + ", momentum: " + str(args.beta))

# Print the equation of the plane found by the neural net.
print("The least-squares regression plane:")
print("MB-GD (batch_size = {0}): ".format(str(args.batchsize)), end='')
print("y = {0:.3f} + {1:.3f} x1 + {2:.3f} x2".format(w[0], w[1], w[2]))

# Print equation of plane found using linear algebra to solve normal equations.
print("Linear algebraic solution: ", end='')
print("y = " + "{0:.3f} + {1:.3f} x1 + {2:.3f} x2". \
      format(lin_alg_sol[0], lin_alg_sol[1], lin_alg_sol[2]))
