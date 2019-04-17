# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os, argparse, pickle

from tqdm import tqdm
from datetime import datetime
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchvision import datasets, transforms
from logger import Logger
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from math import sqrt

# from tensorboardX import SummaryWriter

# local files
from models.cnn_models import *
from utils.unpickle import gather_chromosome_data


def run_on_test(model, loss_fn,  device, test_loader, output_file_path, unsqueeze_data=True):
    """
    Runs model on testing set, saves output in output_file_path
    """
    model.eval()
    preds = []
    mse = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):  # iterates through batches of 64 arrays
            target = data[1]
            data = data[0]
            if unsqueeze_data:
                data = data.unsqueeze(1)
            data = data.to(device)
            output = model(data)  # make classification
            mse += loss_fn(output, target)
            for batch_element in range(test_loader.batch_size):
                preds.append(f"{batch_element + ( i * test_loader.batch_size )},{output[ batch_element ].item()}")

    mse /= ( test_loader.batch_size * len( test_loader ) )
    mse = mse.item()
    mae = sqrt(mse)

    with open(output_file_path, 'w') as predictions_file:
        predictions_file.write('MSE: ' + str(mse))
        predictions_file.write('\nMAE: ' + str(mae))
        predictions_file.write('\nId,Prediction\n')
        predictions_file.write('\n'.join(preds))

    print('MSE: ' + str(mse))
    print('\nMAE: ' + str(mae))
    print(f">>> Wrote predictions in {output_file_path} \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a PyTorch model on the test dataset')
    parser.add_argument('--path-to-model-savefile', type=str, required=True,
                        help="path to the model save file, should be in /pickled-params/<something>/<timestamp>_model.savefile")
    parser.add_argument('--test-data-path', type=str, required=True,
                        help="path to the models test set")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for model (default: 1)')
    parser.add_argument("--RNN", type=bool, default=False,
                        help="should be False for CNN and true for RNN")
    parser.add_argument("--data-transform", type=str, default="None",
                        help="transform to apply to the data before passing to the CNNs (can be 'mult_cap' or 'log').")

    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(">>> Have you checked that the model you are using is the same model as the one(s) you trained with?")
    # useful reference for debugging
    # batch_size = 64
    # path_to_model_savefile = os.path.abspath( r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\mini-project-3\comp551-a3\pickled-params\2019-03-13_22-33_model.savefile" )
    # hard-coded parameters end here

    test_inputs, test_targets, _ = gather_chromosome_data(args.test_data_path)
    print(">>> Loaded test dataset")

    if args.data_transform == "mult_cap":
        # scale by downsample ratio
        print( "\n>>> Multiplying arrays (not targets) by 16.0 and capping arrays and targets at 100.0\n" )
        test_inputs = [ np.clip( arr*16.0, 0.0, 100.0 ) for arr in test_inputs ]
        test_targets = [ max( min( target, 100.0 ), 0.0 ) for target in test_targets ]
    elif args.data_transform == "log":
        print( "\n>>> Appling log( arr + 1.0 ) transform\n" )
        test_inputs = [ np.log( arr+1.0 ) for arr in test_inputs ]
        test_targets = np.log( np.array( test_targets ) + 1.0 )



    print(">>> Preprocessed test dataset images")

    test_tensor_dataset = torch.tensor(test_inputs).double()
    print(f">>> {test_tensor_dataset.shape}")
    test_tensor_targets = torch.tensor(test_targets).double()
    test_tensor_dataset = TensorDataset( test_tensor_dataset, test_tensor_targets )

    test_loader = torch.utils.data.DataLoader(
        test_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )

    start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # load learned parameters
    # hard-coded parameters start here
    model = ThreeLayerModel13(batch_size=args.batch_size).to(device).double()  # casting it to double because of some  weird pytorch peculiarities
    # hard-coded parameters end here
    #if torch.cuda.is_available():
    #    print('Loading model on to GPU')
    #    model.load_state_dict(torch.load(args.path_to_model_savefile))  # gpu use by default
    else:
        print('Loading model on to CPU')
        model.load_state_dict(
            torch.load(args.path_to_model_savefile, map_location='cpu'))  # prevents bugging out from no cuda availability

    print(">>> Loaded model\n\n")

    output_file_path = os.path.join(os.path.dirname(args.path_to_model_savefile),
                                    f'{start_timestamp}_test_set_predictions.csv')
    loss_fn = nn.MSELoss()

    print(">>> Evaluating on test dataset")
    run_on_test(model, loss_fn, device, test_loader, output_file_path, unsqueeze_data=(not args.RNN))

    print(">>> Finished")


