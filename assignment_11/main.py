import torch
import torch.nn as nn
from skimage import io
import du.lib as dulib

torch.cuda.is_available = lambda: False

# read in all the digits
digits = io.imread('numbers.png')
xss = torch.Tensor(5000, 400)
idx = 0

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
    yss[i] = i // 500

xss, xss_means = dulib.center(xss)
xss, xss_stds = dulib.normalize(xss)

indices = torch.randperm(len(xss))
xss = xss[indices]
yss = yss[indices]  # coherently randomize the data
xss_train = xss[:4000]
yss_train = yss[:4000]
xss_test = xss[4000:]
yss_test = yss[4000:]


class LogSoftmaxModel(nn.Module):

    def __init__(self):
        super(LogSoftmaxModel, self).__init__()
        self.layer1 = nn.Linear(400, 300)
        self.layer2 = nn.Linear(300, 200)
        self.layer3 = nn.Linear(200, 100)
        self.layer4 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        return torch.log_softmax(x, dim=1)


model = LogSoftmaxModel()

criterion = nn.NLLLoss()

model = dulib.train(
    model=model,
    crit=criterion,
    train_data=(xss_train, yss_train),
    valid_data=(xss_test, yss_test),
    learn_params={"lr": 0.00025, "mo": 0.952},
    bs=25,
    epochs=100,
    graph=1
)

pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=True)
print(f"Percentage correct on training data: {100 * pct_training:.2f}")

pct_testing = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)
print(f"Percentage correct on testing data: {100 * pct_testing:.2f}")
