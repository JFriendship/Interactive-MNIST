import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvNetMNIST(nn.Module):
    def __init__(self, first_dropout=0.4 , second_dropout=0.2, l1=50):
        super(ConvNetMNIST, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.drop1 = nn.Dropout(p=first_dropout)
        self.drop2 = nn.Dropout(p=second_dropout)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
