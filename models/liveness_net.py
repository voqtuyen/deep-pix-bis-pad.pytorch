import torch
from torch import nn


class LivenessNet(nn.Module):
    def __init__(self):
        super(LivenessNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
        )

        self.layer6 = nn.Conv2d(25, 1, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, x):
        output_layer_1 = self.layer1(x)
        output_layer_2 = self.layer2(output_layer_1)
        output_layer_3 = self.layer3(output_layer_2)
        output_layer_4 = self.layer4(output_layer_3)
        shortcut_1 = self.shortcut(output_layer_2)
        output_immediate = shortcut_1 + output_layer_3 + output_layer_4
        output_layer_5 = self.layer5(output_immediate)
        output_layer_6 = self.layer6(output_layer_5)

        return output_layer_6

