import torch

#Part 2

xs = torch.randn(30)
print(f"{xs}\n"
      f"Standard Deviation: {torch.std(xs)}\n"
      f"Mean: {torch.mean(xs)}")

#Part 3

xs = torch.normal(100, 25, size=(1, 30))
print(f"{xs}\n"
      f"Standard Deviation: {torch.std(xs)}\n"
      f"Mean: {torch.mean(xs)}")

#Part 4

tensorList = []
tensorMean = 0
for i in range(100):
    tensorList.append(torch.normal(100, 25, size=(1, 30)))
    tensorMean += torch.mean(tensorList[i])
tensorMean /= 100
print(f"Mean of a {len(tensorList)} tensors is: {tensorMean}")

# #Part 5

tensorList = []
tensorSTD = 0
for i in range(100):
    tensorList.append(torch.normal(100, 25, size=(1, 30)))
    tensorSTD += torch.std(tensorList[i])
tensorSTD /= 100
print(f"Standard deviation of a {len(tensorList)} tensors is: {tensorSTD}")

#Part 6

tensorList = []
tensorMean = 0
tensorSTD = 0
for i in range(100):
    tensorList.append(torch.randn(30))
    tensorMean += torch.mean(tensorList[i])
    tensorSTD += torch.std(tensorList[i])
tensorMean /= 100

print(f"Mean of a {len(tensorList)} tensors is: {tensorMean}")
print(f"Standard deviation of a {len(tensorList)} tensors is: {tensorSTD}")