# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os, argparse, pickle

from tqdm import tqdm
from datetime import datetime
from collections import Counter
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from logger import Logger


# local files
from models.cnn_models import *
from models.srdensenet import Net as SRDenseNet
from utils.unpickle import gather_chromosome_data


def train(args, model, loss_fn, device, train_loader, optimizer, epoch, minibatch_size, logger):
    model.train()
    outputs, targets = None, None
    for batch_idx, (data, target) in enumerate(train_loader):
        # unsqueeze(x) adds a dimension in the xth-position from the left to deal with the Channels argument of the Conv2d layers
        data = data.unsqueeze( 1 )
        data, target = data.to(device), target.to(device)        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            outputs = output.clone().detach().data.numpy()
            targets = target.clone().detach().data.numpy()
        else:
            outputs = np.vstack( ( outputs, output.clone().detach().data.numpy() ) )
            targets = np.hstack( ( targets, target.clone().detach().data.numpy() ) )


        if ( batch_idx ) % args.log_interval == 0:

            # compute current training accuracy; this is basically the same as loss now
            with torch.no_grad():  # so to not fuck up gradients; i think this is now unnecessary but fine for now

                train_acc = torch.sqrt(loss)

                print('Training Epoch: {} [{}/{} ({:.0f}%)]\t\tTrain Loss: {:.6f}\tTrain Accuracy:{:.1f}\n'.format(
                    epoch, batch_idx+1, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), train_acc ) )


                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {
                    'train loss': loss.item(), 'train accuracy': train_acc
                }

                step = batch_idx + ( epoch *  len( train_loader ) )
                print( f">>> Step:{step}\n" )
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary), commented out manually
                '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

                for tag, images in info.items():
                    logger.image_summary(tag, images, step)'''

    return outputs, targets


def validate(args, model, loss_fn, device, validation_loader, epoch, logger, validation_split_fraction ):
    model.eval()
    validation_loss = 0
    correct = 0
    outputs, targets = None, None 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data = data.unsqueeze(1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += loss_fn(output, target).item()  # sum up batch loss

            if batch_idx == 0:
                outputs = output.data.numpy()
                targets = target.data.numpy()
            else:
                outputs = np.vstack( ( outputs, output.data.numpy() ) )
                targets = np.hstack( ( targets, target.data.numpy() ) )

    validation_loss /= ( validation_loader.batch_size * len( validation_loader ) )
    # recall that notion of accuracy is weird for regression
    accuracy = np.sqrt(validation_loss)

    print('\nValidation set:\t\tAverage loss: {:.4f}, Accuracy: ({:.1f})\n'.format(
        validation_loss, accuracy))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # 1. Log scalar values (scalar summary)
    info = { 'val loss': validation_loss, 'val accuracy': accuracy }
    
    step = epoch * ( len( validation_loader.dataset ) * ( 1. - validation_split_fraction ) ) / validation_loader.batch_size
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)
    
    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

    return np.array(outputs), targets


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Used to run CNN')
    parser.add_argument('--train-data-path', type=str, required=True, metavar='T',
                        help="path to the file containing the pickled list of (array, target, chromosome number) tuples for training")
    parser.add_argument('--valid-data-path', type=str, required=True, metavar='V',
                        help="path to the file containing the pickled list of (array, target, chromosome number) tuples for testing")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--l2', type=float, default=0, metavar='N',
                        help='weight_decay parameter sent to optimizer (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='boolean indicator of verbosity')
    parser.add_argument("--train-chroms", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='list of chromosomes to use as training data')
    parser.add_argument("--valid-chroms", nargs="*", type=int, default=[13],
                        help='list of chromosomes to use as validation data')
    args = parser.parse_args()
        
    # Device configuration
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    
    # Dataset loading
    print( "\n>>> Loading datasets\n" )

    # data_arrays_list, data_targets_list, data_chromnum_list = unpickle_data_pickle( args.data_pickle_path )

    X_valid, y_valid, _ = gather_chromosome_data(args.valid_data_path)
    X_train, y_train, _ = gather_chromosome_data(args.train_data_path)

    train_tensor_dataset = torch.tensor( X_train ).double()
    train_tensor_labels = torch.tensor( y_train ).double() # try .long() if you get a bug

    valid_tensor_dataset = torch.tensor( X_valid ).double()
    valid_tensor_labels = torch.tensor( y_valid ).double() # try .long() if you get a bug

    if args.verbose:
        print( ">>> Loaded datasets\n" )
        
    print( train_tensor_dataset.shape )
    print( train_tensor_labels.shape )
    
    train_tensor_dataset = TensorDataset( train_tensor_dataset, train_tensor_labels )
    valid_tensor_dataset = TensorDataset( valid_tensor_dataset, valid_tensor_labels )
    # assert len( data_arrays_list ) == len( tensor_dataset )

    train_tensor_dataloader = DataLoader(
        train_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    valid_tensor_dataloader = DataLoader(
        valid_tensor_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )


    if args.verbose:
        print( ">>> Compiled tensor dataset" )

    start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )

    logpath = os.path.join( os.getcwd(), 'tensorboard-logs', start_timestamp )

    if not os.path.isdir( logpath ):
        os.makedirs( logpath )
    
    logger = Logger( logpath )

    if args.verbose:
        print( f"\nThe log file will be saved in {logpath.__str__()}\n")

    # Model definition
    model = ThreeLayerModel13(batch_size=args.batch_size).to( device ).double() # casting it to double because of some pytorch expected type peculiarities
    
    # Loss and optimizer
    # parametize this
    optimizer = torch.optim.Adam(  
        model.parameters(),
        lr=args.lr,
        # momentum=args.momentum
    )

    criterion = nn.MSELoss() 

    print( "\n>>> Starting training\n" )
    
    # dummy declarations, get overwritten at the final epoch to contain 
    all_models_final_outputs, all_corresponding_targets = None, None 
      
    for epoch in range( args.epochs ):

        training_output, training_targets = train( args, model, criterion, device, train_tensor_dataloader, optimizer, epoch, args.batch_size, logger )
        validation_split_fraction = 0.2  # todo: remove this?????
        validating_output, validating_targets = validate(args, model, criterion, device, valid_tensor_dataloader, epoch, logger, validation_split_fraction)

        if epoch == ( args.epochs - 1 ): # we only care about the last epoch's output
            all_models_final_outputs = training_output
            all_corresponding_targets = training_targets, 
        
    # Saving output
    if (args.save_model):
  
        torch.save( model.state_dict(), os.path.join( os.getcwd(), 'pickled-params', start_timestamp+'_model.savefile' ) )
    
        if args.merge_train_validate_outputs:
            pickle_path = os.path.join( os.getcwd(), 'pickled-params', start_timestamp+"_merged_training_and_validating_outputs_and_targets.pickle" )
            
            with open( pickle_path, 'wb' ) as meta_clf_feature_mat:
                pickle.dump( np.hstack( ( all_models_final_outputs, all_corresponding_targets.reshape( -1, 1 ) ) ) , meta_clf_feature_mat, protocol=pickle.HIGHEST_PROTOCOL )

            print( f">>> The <features from training & validating><labels from training & validating> matrix pickle for meta-classification is saved under\n{pickle_path}" )
        
        else:                
            training_and_validating_outputs =  np.hstack( ( all_models_final_outputs, all_corresponding_targets.reshape( -1, 1 ) ) )

            training_indices = range( 0, int( len( train_tensor_dataloader.dataset ) * ( 1.0 - args.validation_split_fraction ) ) )
            validating_indices = range( max( training_indices )+1, len( valid_tensor_dataloader.dataset ) )

            for m in range( args.n_models ):
                column_range = list( range( m*10, (m+1)*10 ) )
                column_range.append( training_and_validating_outputs.shape[1]-1 )
                output_subarray_from_training = training_and_validating_outputs[ training_indices, : ]
                output_subarray_from_training = output_subarray_from_training[ :, column_range ]
                
                output_subarray_from_validating = training_and_validating_outputs[ validating_indices, : ]
                output_subarray_from_validating = output_subarray_from_validating[ :, column_range ]

                pickle_path = os.path.join( os.getcwd(), 'pickled-params', start_timestamp+f"_training_outputs_and_targets.pickle" )
                with open( pickle_path, 'wb' ) as handle:
                    pickle.dump( output_subarray_from_training , handle, protocol=pickle.HIGHEST_PROTOCOL )
                
                print( f">>> The <features from training><labels from training> matrix pickle for meta-classification is saved under\n{pickle_path}" )

                pickle_path = os.path.join( os.getcwd(), 'pickled-params', start_timestamp+f"_validating_outputs_and_targets.pickle" )
                with open( pickle_path, 'wb' ) as handle:
                    pickle.dump( output_subarray_from_validating , handle, protocol=pickle.HIGHEST_PROTOCOL )
                
                print( f">>> The <features from validating><labels from validating> matrix pickle for meta-classification is saved under\n{pickle_path}" )

        with open( os.path.join( os.getcwd(), 'pickled-params', start_timestamp+'_params.savefile' ), 'w' ) as params_file:
            params_file.write( args.__repr__() )
            params_file.write( '\n' )
            params_file.write( optimizer.__repr__() )
            params_file.write( '\n' )
            params_file.write( criterion.__repr__() )

    print( f"\nThe log file was saved in {logpath.__str__()}\n")
    print( f"\nThe model and parameter save files were saved in { os.path.join( os.getcwd(), 'pickled-params' ) }\n" )
