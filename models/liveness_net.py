import torch
from torch import nn


class LivenessNet(nn.Module):
    def __init__(self):
        super(LivenessNet, self).__init__()
        self.conv0 = nn.Conv2d(6, 32, 3, 1, padding=1)
        self.max_pool0 = nn.MaxPool2d(3, 2, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 25, 3, 1, padding=1)
        self.max_pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(32, 25, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(25, 32, 3, 1, padding=1)
        self.max_pool2 = nn.MaxPool2d(3, 2, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv8 = nn.Conv2d(32, 25, 3, 1, padding=1)
        self.conv9 = nn.Conv2d(25, 32, 3, 1, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv11 = nn.Conv2d(32, 25, 3, 1, padding=1)
        self.conv12 = nn.Conv2d(25, 2, 3, 1, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(25)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(25)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(25)
        self.bn9 = nn.BatchNorm2d(32)
        self.bn10 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(2)


    def forward(self, x):
        x = self.conv0(x)
        return  x
