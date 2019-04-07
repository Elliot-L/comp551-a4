"""
Modified from cnn_trial file from assignment 3
"""

# warning: VSC is weird with torch

from __future__ import print_function

import numpy as np
import argparse, random, torch, torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
# from tensorboardX import SummaryWriter
# local files
from models.cnn_models import *


def main(cli_args, device, logdir=os.path.join(os.getcwd(), 'logs'), shuffle=True, verbose=True):

    # generating some mock data
    training_data = np.array(np.random.randint(low=0, high=10, size=(1, 1, 13, 13)), dtype=np.float32)
    training_labels = np.array(np.random.randint(low=0, high=10, size=(1, 1)), dtype=np.float32)
    # convert from np to tensors
    training_data = torch.tensor(training_data)
    training_labels = torch.tensor(training_labels)
    tensor_dataset = TensorDataset(training_data, training_labels)

    batch_size = cli_args.batch_size
    if batch_size == 0:
        batch_size = training_data.shape[0]

    tensor_dl = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if verbose:
        print("data has been loaded")

    # model = ThreeLayerModel.to(device)
    model = BaseNet()

    # parametize this
    optimizer = torch.optim.Adam(  # Adam
        model.parameters(),
        lr=args.lr,
        # momentum=args.momentum
    )

    # parametize this
    criterion = torch.nn.MSELoss()

    if verbose:
        print("model, optimizer, and criterion have been defined")

    losses, accuracies = [], []
    if verbose:
        print("starting the run!")

    for epoch in range(cli_args.epochs):

        model.train()
        for batchidx, (data_instance, data_label) in enumerate(tensor_dl):

            # forward pass
            preds = model(data_instance)

            # compute loss and accuracy
            loss = criterion(preds, data_label)
            losses.append(loss)

            accuracies.append(torch.sqrt(loss))  # accuracy is basically same as loss for MSE

            # reset gradients to zero, then do a backward prop and update the weights
            optimizer.zero_grad()  # zeros out any gradient buffers from previous iterations
            loss.backward()
            optimizer.step()

            if batchidx % cli_args.log_interval == 0:
                print(
                    f"training epoch {epoch} / {cli_args.epochs}, batch #{batchidx} / {training_data.shape[0] // batch_size}\nLoss:\t{losses[-1]},\t\tAcc:\t{accuracies[-1]}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running Training Example')

    parser.add_argument('--training-dataset-path', type=str, default="train_images.pkl",
                        help="path to the training dataset pickle file (default: train_images.pkl)")
    parser.add_argument('--training-labels-path', type=str, default="train_labels.csv",
                        help="path to the training labels csv file (default: train_labels.csv)")
    parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                        help='input batch size for training (default: 0, meaning entire dataset)')
    parser.add_argument('--test-batch-size', type=int, default=0, metavar='N',
                        help='input batch size for testing (default: 0, meaning entire dataset)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=10E-03, metavar='LR',
                        help='learning rate (default: 10E-03)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training (default: True)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    main(args, device)

