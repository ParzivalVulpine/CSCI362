import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib

digits = io.imread('numbers.png')
xss = torch.Tensor(5000, 20, 20)
idx = 0
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss[idx] = torch.Tensor((digits[i:i + 20, j:j + 20]))
        idx = idx + 1

yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
    yss[i] = i // 500

indices = torch.randperm(len(xss))
xss = xss[indices]
yss = yss[indices]
xss_train = xss[:4000]
yss_train = yss[:4000]
xss_test = xss[4000:]
yss_test = yss[4000:]

xss_train, xss_means = dulib.center(xss_train)
xss_train, xss_stds = dulib.normalize(xss_train)

xss_test, _ = dulib.center(xss_test, xss_means)
xss_test, _ = dulib.normalize(xss_test, xss_stds)


class ConvolutionalModel(nn.Module):

  def __init__(self):
    super(ConvolutionalModel, self).__init__()
    self.meta_layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    )
    self.meta_layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    self.fc_layer1 = nn.Linear(800,10)

  def forward(self, xss):
    xss = torch.unsqueeze(xss, dim=1)
    xss = self.meta_layer1(xss)
    xss = self.meta_layer2(xss)
    xss = torch.reshape(xss, (-1, 800))
    xss = self.fc_layer1(xss)
    return torch.log_softmax(xss, dim=1)

# create an instance of the model class
model = ConvolutionalModel()

# set the criterion
criterion = nn.NLLLoss()

model = dulib.train(
    model=model,
    crit=criterion,
    train_data=(xss_train, yss_train),
    valid_data=(xss_test, yss_test),
    learn_params={"lr": 0.00021, "mo": 0.956},
    bs=25,
    epochs=100,
    graph=1
)

pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=True)
print(f"Percentage correct on training data: {100 * pct_training:.2f}")

pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)
print(f"Percentage correct on testing data: {100 * pct_testing:.2f}")