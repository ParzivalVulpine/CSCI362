import csv
import torch
import matplotlib.pyplot as plt

# Read the data from the CSV file
with open('assign2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # skip the first line
    xs, ys = [], []
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))

# Convert the data to PyTorch tensors
xsTensor = torch.tensor(xs)
ysTensor = torch.tensor(ys)

# Create the design matrix X with ones in the first column and xs in the second column
X = torch.stack((torch.ones_like(xsTensor), xsTensor), dim=1)

# Calculate the coefficients using the matrix approach w=(X^T * X)^-1 * X^T * y
XT = X.t()  # Transpose of X
XTX = XT.mm(X)  # X^T * X
# X^T * y (note that we need to reshape y to be a column vector)
XTy = XT.mm(ysTensor.view(-1, 1))

# Solve for w using the normal equation
w = torch.inverse(XTX).mm(XTy)

# Extract the intercept and slope from w
intercept, slope = w[0][0], w[1][0]

# Print the regression equation
print(f"Regression Equation: y = {slope.item()}x + {intercept.item()}")

# Define the regression line function


def regression_line(x):
    return slope * x + intercept


# Plot the data and regression line
plt.scatter(xs, ys, label='Data')
plt.plot(xs, regression_line(xsTensor).detach().numpy(),
         color='red', label='Regression Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()
