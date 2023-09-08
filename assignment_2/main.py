import torch
import csv
with open('assign2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile)  # skip the first line
    xs, ys = [], []
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
xs, ys = torch.tensor(xs), torch.tensor(ys)
