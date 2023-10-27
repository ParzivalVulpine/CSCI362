"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
"""

import pandas as pd
import torch
import torch.nn as nn

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice', '1st_Flr_SF', '2nd_Flr_SF', 'Lot_Area', 'Overall_Qual',
         'Overall_Cond', 'Year_Built', 'Year_Remod/Add', 'BsmtFin_SF_1', 'Total_Bsmt_SF',
         'Gr_Liv_Area', 'TotRms_AbvGrd', 'Bsmt_Unf_SF', 'Full_Bath']
df = pd.read_csv('AmesHousing.csv', names=names)
data = df.values  # read data into a numpy array (as a list of lists)
data = data[1:]  # remove the first list which consists of the labels
data = data.astype(float)  # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data)  # convert data to a Torch tensor

data.sub_(data.mean(0))  # mean-center
data.div_(data.std(0))  # normalize

xss = data[:, 1:]
yss = data[:, :1]


# define a model class


class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        # Modified layer 1 to become the output for layer 2
        self.layer1 = nn.Linear(13, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, xss):
        # Added relu activation function
        xss = self.layer1(xss)
        xss = torch.relu(xss)
        return self.layer2(xss)


# create and print an instance of the model class
model = LinearModel()
print(model)

criterion = nn.MSELoss()

num_examples = len(data)
batch_size = 25
learning_rate = .007
momentum = .9
epochs = 1000

# Implementing momentum by hand
z_parameters = []
for param in model.parameters():
    z_parameters.append(param.data.clone())
for param in z_parameters:
    param.zero_()

# train the model
for epoch in range(epochs):
    for _ in range(num_examples // batch_size):
        # randomly pick batchsize examples from data
        indices = torch.randperm(num_examples)[:batch_size]

        yss_mb = yss[indices]  # the targets for the mb (minibatch)
        yhatss_mb = model(xss[indices])  # model outputs for the mb

        loss = criterion(yhatss_mb, yss_mb)
        model.zero_grad()
        loss.backward()  # back-propagate

        # update weights
        for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
            z_parameters[i] = momentum * z_param + param.grad.data
            param.data.sub_(z_parameters[i] * learning_rate)

    with torch.no_grad():
        total_loss = criterion(model(xss), yss).item()

    print_str = "epoch: {0}, loss: {1}".format(
        epoch + 1, total_loss * batch_size / num_examples)
    if epochs < 20 or epoch < 7 or epoch > epochs - 17:
        print(print_str)
    elif epoch == 7:
        print("...")
    else:
        print(print_str, end='\b' * len(print_str), flush=True)

print(f"total number of examples: {num_examples}")
print(f"batch size: {batch_size}")
print(f"learning rate: {learning_rate}")
print(f"momentum: {momentum}")

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
SS_E = 0.0
SS_T = 0.0
mean = yss.mean()
for xs, ys in zip(xss, yss):
    SS_E = SS_E + (ys - model(xs)) ** 2
    SS_T = SS_T + (ys - mean) ** 2
print(f"1-SSE/SST = {1.0 - (SS_E / SS_T).item():1.4f}")