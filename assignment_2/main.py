import torch
import csv
import matplotlib.pyplot as plt
with open('assign2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # skip the first line
    xs, ys = [], []
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
xs, ys = torch.tensor(xs), torch.tensor(ys)

regTensor = torch.empty(1, 60)
for i in range(60):
    regTensor[1][i] = i
    regTensor[0][i] = 1
print(regTensor)

plt.scatter(xs.numpy(), ys.numpy())
plt.show()
