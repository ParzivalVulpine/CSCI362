import du.lib as dulib
import pandas as pd
import torch
import torch.nn as nn

# Read the named columns from the csv file into a dataframe.
names = [
    "SalePrice",
    "1st_Flr_SF",
    "2nd_Flr_SF",
    "Lot_Area",
    "Overall_Qual",
    "Overall_Cond",
    "Year_Built",
    "Year_Remod/Add",
    "BsmtFin_SF_1",
    "Total_Bsmt_SF",
    "Gr_Liv_Area",
    "TotRms_AbvGrd",
    "Bsmt_Unf_SF",
    "Full_Bath",
]
df = pd.read_csv("AmesHousing.csv", names=names)
data = df.values  # read data into a numpy array (as a list of lists)
data = data[1:]  # remove the first list which consists of the labels
data = data.astype(float)  # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data)  # convert data to a Torch tensor

data.sub_(data.mean(0))  # mean-center
data.div_(data.std(0))  # normalize

xss = data[:, 1:]
yss = data[:, :1]

# split the data into training and testing sets
random_split = torch.randperm(len(xss))
xss_training = xss[random_split[:2000]]
yss_training = yss[random_split[:2000]]
xss_testing = xss[random_split[2001:-1]]
yss_testing = yss[random_split[2001:-1]]

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

num_examples = len(xss_training)
batch_size = 10
learning_rate = 0.007
momentum = 0.9
epochs = 125

model = dulib.train(
    model=model,
    crit=criterion,
    train_data=(xss_training, yss_training),
    valid_data=(xss_testing, yss_testing),
    learn_params={"lr": learning_rate, "mo": momentum},
    bs=batch_size,
    epochs=epochs,
    graph=1,
)

# Implementing momentum by hand
z_parameters = []
for param in model.parameters():
    z_parameters.append(param.data.clone())
for param in z_parameters:
    param.zero_()

print(f"total number of examples: {num_examples}")
print(f"batch size: {batch_size}")
print(f"learning rate: {learning_rate}")
print(f"momentum: {momentum}")

print(
    f"explained variation: {dulib.explained_var(model, (xss_training, yss_training))}"
)
print(
    f"explained variation (in test dataset): "
    f"{dulib.explained_var(model, (xss_testing, yss_testing))}"
)
