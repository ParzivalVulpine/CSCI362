import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib

torch.cuda.is_available = lambda : False

# read in all of the digits
digits = io.imread('numbers.png')
xss = torch.Tensor(5000,400)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

# # extract just the zeros and eights from xss
# tempxss = torch.Tensor(1000,400)
# tempxss[:500] = xss[:500]
# tempxss[500:] = xss[4000:4500]

# # overwrite the original xss with just zeros and eights
# xss = tempxss

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500

xss, xss_means = dulib.center(xss)
xss, xss_stds = dulib.normalize(xss)

class LogSoftmaxModel(nn.Module):

    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        self.layer1 = nn.Linear(400,200)
        self.layer2 = nn.Linear(200,10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return torch.log_softmax(x, dim=1)

model = LogSoftmaxModel()

criterion = nn.NLLLoss()

def pct_correct(yhatss, yss):
  count = 0
  for i in range(len(xss_test)):
    if  torch.argmax(model(xss_test[i].unsqueeze(0))).item() == yss_test[i].item():
      count += 1
  print("Percentage correct on test set:",100*count/len(xss_test))

model = dulib.train(
  model=model,
  crit=criterion,
  train_data=(xss, yss),
  learn_params={"lr": 0.007, "mo":0.9},
  bs=10,
  valid_metric=pct_correct,
  epochs=150,
  graph=1
)

print("Percentage correct:",pct_correct(model(xss), yss))
