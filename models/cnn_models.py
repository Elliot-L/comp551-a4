import argparse, os, torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
RNN model
[169] -> [1] i.e. takes in flattened [13x13] input
"""

class RNN13(nn.Module):
    def __init__(self, num_hidden=13, num_layers=2, batch_size=1):
        super(RNN13, self).__init__()
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.rnn = nn.LSTM(input_size=13, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        out_rnn, h_rnn = self.rnn(x, None)
        x = out_rnn.view(self.batch_size, self.num_hidden)
        x = self.fc1(x[:, -1, :])
        return x


"""
[13x13] -> [1]
Simple network to predict the center value of a 13x13 input
"""
class ThreeLayerModel13(nn.Module):
    # Modified from: https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        super(ThreeLayerModel13, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    # comments for shapes are assuming batch size of 1
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  # [1, 1, 13, 13] -> [1, 16, 7, 7]
        x = F.max_pool2d(x, 2, 2)  # [1, 16, 7, 7] -> [1, 16, 3, 3]
        x = F.relu(self.conv2_bn(self.conv2(x)))  # [1, 16, 3, 3] -> [1, 64, 2, 2]
        x = F.relu(self.conv3(x))  # [1, 64, 2, 2] -> [1, 128, 1, 1]
        x = x.view(self.batch_size, 128)  # [1, 128, 1, 1] -> [1, 128]
        x = F.relu(self.fc1(x))  # [1, 128] -> [1, 32]
        x = self.fc2(x)  # [1, 32] -> [1, 1]
        return x

"""
[13x13] -> [1]
or
[40x40]->[28x28]
Base network from paper
"""
class BaseNet(nn.Module):
    # Taken from https://github.com/zhangyan32/HiCPlus_pytorch/blob/master/src/model.py (paper author)
    # very slight modifications, same functionality
    def __init__(self, batch_size=1, D_in=None, D_out=None):  # changed args to default values (unused)
        self.batch_size = batch_size
        conv2d1_filters_numbers = 8
        conv2d1_filters_size = 9
        conv2d2_filters_numbers = 8
        conv2d2_filters_size = 1
        conv2d3_filters_numbers = 1  # seems 1 is directly passed in instead of using this
        conv2d3_filters_size = 5
        super(BaseNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x



"""
[40x40]->[28x28]
Base network from paper with minor improvements
"""
class BaseNetPlus(nn.Module):
    # Taken from https://github.com/zhangyan32/HiCPlus_pytorch/blob/master/src/model.py (paper author)
    # Modifications for batchnorm + removing relu from the output layer
    def __init__(self, batch_size=1, D_in=None, D_out=None):  # changed args to default values (unused)
        self.batch_size = batch_size
        conv2d1_filters_numbers = 8
        conv2d1_filters_size = 9
        conv2d2_filters_numbers = 8
        conv2d2_filters_size = 1
        conv2d3_filters_numbers = 1  # seems 1 is directly passed in instead of using this
        conv2d3_filters_size = 5
        super(BaseNetPlus, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv1_bn = nn.BatchNorm2d(conv2d1_filters_numbers)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv2_bn = nn.BatchNorm2d(conv2d1_filters_numbers)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)


    def forward(self, x):
        #print("start forwardingf")
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x




"""
[40x40]->[28x28]
Base network from paper with minor improvements
"""
class ThreeLayerModel40(nn.Module):
    def __init__(self, batch_size=1):
        super(ThreeLayerModel40, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152, 1024)
        self.fc2 = nn.Linear(1024, 784)

    # comments for shapes are assuming batch size of 1
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x))) # [1, 1, 40, 40] -> [1, 16, 20, 20]
        x = F.max_pool2d(x, 2, 2) # [1, 16, 20, 20] -> [1, 16, 10, 10]
        x = F.relu(self.conv2_bn(self.conv2(x))) # [1, 16, 10, 10] -> [1, 64, 9, 9]
        x = F.max_pool2d(x, 2, 2) # [1, 64, 9, 9] -> [1, 64, 4, 4]
        x = F.relu(self.conv3_bn(self.conv3(x)))  # [1, 64, 4, 4] -> [1, 128, 3, 3]
        x = x.view(self.batch_size, 1152) # [1, 128, 3, 3] -> [1, 1152]
        x = F.relu(self.fc1(x)) # [1, 1152] -> [1, 1024]
        x = self.fc2(x) # [1, 1024] -> [1, 784]
        x = x.reshape(self.batch_size, 1, 28, 28)
        return x
