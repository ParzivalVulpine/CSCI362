#!/usr/bin/env python3
# ols_tensor.py                                                SSimmons Feb 2018
"""
This is an implementation, using only tensor arithmetic, of a neural net that
finds the ordinary least-squares regression plane. It learns via (batch)
gradient descent. The data is first mean-centered and normalized.

Notes:
  - The climate data consist of 32 examples, each with 2 features (inputs)
    and 1 target (output); upon training, the model learns the regression plane
    that best fits the data in the sense that it minimizes the sum of the
    squares of the residuals.
  - The features of a single example are denoted by xs; the set of all features
    is denoted xss; similarly, the set of all targets is denoted yss, and a
    single target is denoted ys (even though the targets for the climate data
    consist of only a single feature).
  - After printing out the equation of the learned regression plane, the least-
    squares regression plane found linearly algebraically (by solving
    the normal equations) is displayed, for comparison.
  - To find the least-squares regression plane linear-algebraically, we could
    solve the normal equations as we did in Project 0. Below we instead use the
    built-in gels library, which is presumably numerically robost.
    The reason that numerical considerations are in play is that the computa-
    tions we need to make involve for example inverting a matrix. And since we
    we don't pre-center and normalize (see the code below) the matrices may be
    numerically ill-conditioned.
"""
import csv
import torch

# This code block reads the data from the csv file and, skipping the first line,
# writes the 2nd, 3rd, and 4th elements of each line to appropriate lists.
with open('temp_co2_data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # Skip the first line of csvfile.
    # Create empty lists to hold the features and the targets.
    xss, yss = [], []
    for row in reader:
        # The 3rd and 4th entries are the features of each example.
        xss.append([float(row[2]), float(row[3])])
        # The 2nd entry of each line is the corresponding output.
        yss.append([float(row[1])])

# Convert xss and yss to torch tensors.
xss = torch.tensor(xss)  # xss is now a 32x2 tensor:  torch.Size([32,2])
yss = torch.tensor(yss)  # yss is now a 32x1 tensor:  torch.Size([32,1])

# For comparison purposes at the end of this program, compute (before we center
# and normalize) the least squares regression plane using linear algebra.
A = torch.cat((torch.ones(len(xss), 1), xss), 1)
lin_alg_sol = torch.linalg.lstsq(A, yss, driver='gels').solution[:, 0]

# Compute the column-wise means and standard deviations.
# Comment out the next 2 lines, for example, to see what happens if you do not
# mean center; or comment out all 4 lines; or just the last 2.
xss_means = xss.mean(0)  # xss_means.size() returns torch.size([2])
yss_means = yss.mean()   # yss_means.size() returns torch.size([])
xss_stds = xss.std(0)   # similarly here
yss_stds = yss.std()    # and here

# Mean center the inputs and output (if xss_means and yss_means are defined).
if 'xss_means' in locals() and 'yss_means' in locals():
    xss, yxx = xss - xss_means, yss - yss_means

# Normalize the inputs and output (if xss_stds and yss_stds are defined).
if 'xss_stds' in locals() and 'yss_stds' in locals():
    xss, yss = xss/xss_stds, yss/yss_stds

# Concatenate a column of 1's onto the beginning of the features tensor xss so
# as to implement the bias.
xss = torch.cat((torch.ones(len(xss), 1), xss), 1)  # torch.Size([32,3])

# The weights, randomly initialized. Here, w[0] will be the bias.
w = torch.rand(3, 1)-0.5*torch.ones(3, 1)  # torch.Size([3,1])

alpha = 0.5  # learning rate
epochs = 30  # the total number of times the model sees all of the data
num_examples = len(xss)  # the number of examples

for epoch in range(epochs):  # the training loop

    # Compute the estimates of the targets all at once by feeding forward using
    # matrix multiplication.
    yss_pred = xss.mm(w)  # torch.Size([32,1])

    # Compute and print the current loss, which is the mean squared error:
    #     sum_{k=1}^32 (yss_pred[k] - yss[k])^2 / 32.
    loss = (yss_pred - yss).pow(2).sum()/num_examples  # torch.Size([])
    print("epoch: {0}, current loss: {1}".format(epoch+1, loss.item()))

    # Compute the gradient of the loss function w/r to the weights which is the
    # vector with jth component (for j=0,1,2) equal to:
    #    2 * sum_{k=1}^32 (yss_pred[k] - yss[k]) * xss[k][j] / 32.
    grad = 2*((yss_pred-yss)*xss).sum(0, True).t() / \
        num_examples  # torch.Size([3,1])

    # update the weights.
    w = w - alpha * grad  # torch.Size([3,1])

# Squeeze away a dimension.
w = w.squeeze(1)  # torch.Size([3])

# Un-mean-center and un-normalize (trivially, if not necessary).
if not ('xss_means' in locals() and 'yss_means' in locals()):
    xss_means = torch.Tensor([0., 0.])
    yss_means = 0.0
    print('\nno centering')
if not ('xss_stds' in locals() and 'yss_stds' in locals()):
    xss_stds = torch.Tensor([1., 1.])
    yss_stds = 1.0
    print('no normalization')
w[1:] = w[1:] * yss_stds / xss_stds
w[0] = w[0] * yss_stds + yss_means - w[1:] @ xss_means   # @ is the dot product

print("The least-squares regression plane:")
# Print out the equation of the plane found using gradient descent.
print("  found by the neural net is: "+"y = {0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"
      .format(w[0], w[1], w[2]))

# Print out the eq. of the plane found using closed-form linear alg. solution.
print("  using linear algebra:       y = "+"{0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"
      .format(lin_alg_sol[0], lin_alg_sol[1], lin_alg_sol[2]))
print("learning rate:", alpha)
