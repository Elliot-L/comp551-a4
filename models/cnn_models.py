import argparse, os, torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
Input for models is a 13x13 matrix
Return is a single value
"""


class ThreeLayerModel(nn.Module):
    # Modified from: https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self):
        super(ThreeLayerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  # [1, 1, 13, 13] -> [1, 16, 7, 7]
        x = F.max_pool2d(x, 2, 2)  # [1, 16, 7, 7] -> [1, 16, 3, 3]
        x = F.relu(self.conv2_bn(self.conv2(x)))  # [1, 16, 3, 3] -> [1, 64, 2, 2]
        x = F.relu(self.conv3(x))  # [1, 64, 2, 2] -> [1, 128, 1, 1]
        x = x.view(1, 128)  # [1, 128, 1, 1] -> [1, 128]
        x = F.relu(self.fc1(x))  # [1, 128] -> [1, 32]
        x = self.fc2(x)  # [1, 32] -> [1, 1]
        return x
