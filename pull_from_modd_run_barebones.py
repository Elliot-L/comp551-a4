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
from utils.unpickle import unpickle_data_pickle


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
                    100. * batch_idx / len(train_loader), loss.item(), train_acc.item() ) )
                

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {    
                    'train loss': loss.item(), 'train accuracy': train_acc.item()
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


def validate(args, model, loss_fn, device, validation_loader, epoch, logger):
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
    
    accuracy = torch.sqrt(validation_loss).item()

    print('\nValidation set:\t\tAverage loss: {:.4f}, Accuracy: ({:.1f})\n'.format(
        validation_loss, accuracy))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # 1. Log scalar values (scalar summary)
    info = { 'val loss': validation_loss, 'val accuracy': accuracy }

    step = ( epoch + 1 ) * len( validation_loader.dataset ) / validation_loader.batch_size
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
    parser.add_argument( '--data-pickle-dir-path', type=str, required=True, metavar='P',
                        help="path to the directory containing each chromosome's pickleld list of (array, target, chromosome number) tuples." )
    parser.add_argument( '--use-chroms', type=int, nargs='+', required=True, metavar='C',
                        help="space-delimited list of chromosomes integers (e.g. 1 2 3 for chromosomes 1, 2, and 3) to use for this run." )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--l2', type=float, default=0, metavar='N',
                        help='weight_decay parameter sent to optimizer (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
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
    
    data_arrays_list, data_targets_list, data_chromnum_list = [],[],[]
    files_in_dir = [ os.path.join( args.data_pickle_dir_path, f ) for f in os.listdir( args.data_pickle_dir_path ) if os.path.isfile( os.path.join( args.data_pickle_dir_path, f ) ) and '.pickle' in f ]
    print( f"\n>>> Found {len( files_in_dir )} pickled files in {args.data_pickle_dir_path}\n" )
    for chrnum in args.use_chroms:
        identified_file = None
        for file in files_in_dir:
            if 'chr{}_'.format( chrnum ) in file:
                identified_file = file
                break 
        assert identified_file is not None 

        this_data_arrays_list, this_data_targets_list, this_data_chromnum_list = unpickle_data_pickle( identified_file ) 
        data_arrays_list.extend( this_data_arrays_list )
        data_targets_list.extend( this_data_targets_list )

    tensor_dataset = torch.tensor( data_arrays_list ).double() 
    tensor_labels = torch.tensor( data_targets_list ).double() # try .long() if you get a bug

    if args.verbose:
        print( ">>> Loaded datasets\n" )
        
    print( tensor_dataset.shape )
    print( tensor_labels.shape )

    tensor_dataset = TensorDataset( tensor_dataset, tensor_labels )
    assert len( data_arrays_list ) == len( tensor_dataset ) == len( tensor_labels ) 

    tensor_dataloader = DataLoader(
        tensor_dataset,
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
    # model = BaseNet( ).to( device ).double()
    model = ThreeLayerModel13( batch_size=args.batch_size ).to( device ).double() # casting it to double because of some pytorch expected type peculiarities
    
    # Loss and optimizer
    # parametize this
    optimizer = torch.optim.Adam(  
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.MSELoss() 

    print( "\n>>> Starting training\n" )
    
    # dummy declarations, get overwritten at the final epoch to contain 
    all_models_final_outputs, all_corresponding_targets = None, None 
      
    for epoch in range( args.epochs ):
        training_output, training_targets = train( args, model, criterion, device, tensor_dataloader, optimizer, epoch, args.batch_size, logger )
        validation_split_fraction = 0.2  # todo: remove this?????
        validating_output, validating_targets = validate(args, model, criterion, device, tensor_dataloader, epoch, logger, validation_split_fraction)

        if epoch == ( args.epochs - 1 ): # we only care about the last epoch's output
            all_models_final_outputs = training_output
            all_corresponding_targets = training_targets
        
    # Saving output
    if ( args.save_model ):
        
        if not os.path.isdir( os.path.join( os.getcwd(), 'pickled-model-params' ) ): 
            os.makedirs( os.path.join( os.getcwd(), 'pickled-model-params' ) )

        torch.save( model.state_dict(), os.path.join( os.getcwd(), 'pickled-params', start_timestamp+'_model.savefile' ) )
    
        training_and_validating_outputs =  np.hstack( ( all_models_final_outputs, all_corresponding_targets.reshape( -1, 1 ) ) )

        training_indices = range( 0, int( len( tensor_dataloader.dataset ) * ( 1.0 - args.validation_split_fraction ) ) )
        validating_indices = range( max( training_indices )+1, len( tensor_dataloader.dataset ) )

        with open( os.path.join( os.getcwd(), 'pickled-params', start_timestamp+'_params.savefile' ), 'w' ) as params_file:
            params_file.write( args.__repr__() )
            params_file.write( '\n' )
            params_file.write( optimizer.__repr__() )
            params_file.write( '\n' )
            params_file.write( criterion.__repr__() )

    print( f"\nThe log file was saved in {logpath.__str__()}\n")
    print( f"\nThe model and parameter save files were saved in { os.path.join( os.getcwd(), 'pickled-params' ) }\n" )
