import torch
from torch import nn


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64/2=32

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32/2=16

        self.fc1 = nn.Linear(16384, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 16384)
        x = self.fc1(x)

        x=nn.functional.sigmoid(x)
        return x