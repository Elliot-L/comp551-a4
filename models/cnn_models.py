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
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x))) # [64, 1, 64, 64] -> [64, 16, 32, 32]
        x = F.max_pool2d(x, 2, 2) # [64, 16, 32, 32] -> [64, 16, 16, 16]
        x = F.relu(self.conv2_bn(self.conv2(x))) # [64, 16, 16, 16] -> [64, 64, 8, 8]
        x = F.max_pool2d(x, 2, 2) # [64, 64, 8, 8] -> [64, 64, 4, 4]
        x = F.relu(self.conv3_bn(self.conv3(x)))  # [64, 64, 4, 4] -> [64, 128, 2, 2]
        x = x.view(64, 512) # [64, 64, 4, 4] -> [64, 1024]
        x = F.relu(self.fc1(x)) # [64, 1024] -> [64, 500]  # new shape above
        x = self.fc2(x) # [64, 500] -> [64, 10]  # new shape above
        return x
