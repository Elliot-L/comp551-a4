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

# from tensorboardX import SummaryWriter

# local files
from utils import unpickle
from models.cnn_models import *


def train(args, model, loss_fn, device, train_loader, validation_loader, optimizer, epoch, minibatch_size, logger):
    model.train()
    outputs, targets, original_dataset_indices = None, None, None
    for batch_idx, (data, target) in enumerate(train_loader):
        # unsqueeze(x) adds a dimension in the xth-position from the left to deal with the Channels argument of the Conv2d layers
        data = data.unsqueeze(1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            outputs = output.clone().detach().data.numpy()
            targets = target.clone().detach().data.numpy()
            original_dataset_indices = None
        else:
            outputs = np.vstack((outputs, output.clone().detach().data.numpy()))
            targets = np.hstack((targets, target.clone().detach().data.numpy()))

        if (batch_idx + 1) % args.log_interval == 0:

            # compute current training accuracy
            with torch.no_grad():  # so to not fuck up our gradients
                preds = model(data)
                preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = preds.eq(target.view_as(preds)).sum().item()

                train_acc = 100. * correct / (train_loader.batch_size)

                print('Training Epoch: {} [{}/{} ({:.0f}%)]\t\tTrain Loss: {:.6f}\tTrain Accuracy:{:.1f}%\n'.format(
                    epoch, batch_idx + 1, len(train_loader),
                           100. * batch_idx / len(train_loader), loss.item(), train_acc))

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {
                    'train loss': loss.item(), 'train accuracy': train_acc
                }

                step = batch_idx + (epoch * len(train_loader))
                print(f">>> Step:{step}\n")
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary), commented out manually
                '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

                for tag, images in info.items():
                    logger.image_summary(tag, images, step)'''

    # assert outputs.shape[1] == 10
    # assert targets.ndim == 1
    return outputs, targets


def validate(args, model, loss_fn, device, validation_loader, epoch, logger, validation_split_fraction):
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
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx == 0:
                outputs = output.data.numpy()
                targets = target.data.numpy()
            else:
                outputs = np.vstack((outputs, output.data.numpy()))
                targets = np.hstack((targets, target.data.numpy()))

    validation_loss /= (validation_loader.batch_size * len(validation_loader))
    accuracy = 100. * correct / (validation_loader.batch_size * len(validation_loader))

    print('\nValidation set:\t\tAverage loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, correct, (validation_loader.batch_size * len(validation_loader)),
        accuracy))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # 1. Log scalar values (scalar summary)
    info = {'val loss': validation_loss, 'val accuracy': accuracy}

    step = epoch * (len(validation_loader.dataset) * (1. - validation_split_fraction)) / validation_loader.batch_size
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

    # 3. Log training images (image summary), commented out manually
    '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, step)'''
    # change the output shape as appropriate
    return np.array(outputs).reshape(-1, 1), targets


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Train Model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--l2', type=float, default=0, metavar='N',
                        help='weight_decay parameter sent to optimizer (default: 0.0001)')
    parser.add_argument('--validation-split-fraction', type=float, default=0.2, metavar='V',
                        help='the fraction (0.#) of the training dataset to set aside for validation')
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
    parser.add_argument('--MNIST-sanity-check', type=bool, default=False,
                        help="Whether to run the model on PyTorch's MNIST dataset first")
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='boolean indicator of verbosity')
    parser.add_argument('--n-models', type=int, default=1,
                        help='number of models to train')
    parser.add_argument('--training-loader-pickle', type=str, default=None,
                        help='path to the pickled training loader to use (essential for meta-classification)')
    parser.add_argument('--validating-loader-pickle', type=str, default=None,
                        help='path to the pickled validating loader to use (essential for meta-classification)')
    parser.add_argument('--save-loaders', type=bool, default=False,
                        help='whether to save the training and validating data loaders for re-use (essential for meta-classification)')
    parser.add_argument('--merge-train-validate-outputs', type=bool, default=False,
                        help='whether to join the predictions on the training set with those of the validation set')
    parser.add_argument('--preprocess-images', type=bool, default=False,
                        help='whether to preprocess the images')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset loading
    print("\n>>> Loading datasets\n")
    tensor_dataset = None  # dummy declaration

    print('loading data')
    training_data_raw = load_training_data('train_images.pkl', as_tensor=False)
    # for debugging
    # training_data_raw = load_training_data( 'train_images.pkl', as_tensor=True ).double()

    training_data = torch.tensor(training_data_raw).double()

    # for debugging
    # training_data = training_data_raw
    if args.verbose:
        print(">>> Loaded and cleaned (extracted) training data")

    training_labels = load_training_labels('train_labels.csv', as_tensor=True).long()

    if args.rotate_images:
        training_labels_one_set = load_training_labels('train_labels.csv', as_tensor=False)
        training_labels = torch.tensor(np.hstack((training_labels_one_set, training_labels_one_set,
                                                  training_labels_one_set, training_labels_one_set))).long()

    print(training_data.shape)
    print(training_labels.shape)

    tensor_dataset = TensorDataset(training_data, training_labels)
    assert len(training_labels) == len(training_data)
    if args.verbose:
        print(">>> Compiled tensor dataset")

    # Creating train/validation datasets
    print("\n>>> Splitting datasets\n")
    dataset_size = len(tensor_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split_fraction * dataset_size))
    if args.verbose:
        print(f">>> Dataset size: {dataset_size}")
        print(f">>> Validation split: {args.validation_split_fraction}")
        print(f">>> Number of validation instances: {split}")

    if args.verbose:
        print(f">>> Randomizing dataset prior to splitting (according to args.seed)")

    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader, validation_loader = None, None  # dummy declarations

    if args.training_loader_pickle:
        with open(args.training_loader_pickle, 'rb') as handle:
            train_loader = pickle.load(handle)
        print(f">>> Loaded {args.training_loader_pickle}")
    else:
        train_sampler = SequentialSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler
            # shuffle=True # already shuffled
        )

    if args.validating_loader_pickle:
        with open(args.validating_loader_pickle, 'rb') as handle:
            validation_loader = pickle.load(handle)
        print(f">>> Loaded {args.validating_loader_pickle}")
    else:
        valid_sampler = SequentialSampler(val_indices)

        validation_loader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler
            # shuffle=True # already shuffled
        )

    if (args.training_loader_pickle is None) and (args.validating_loader_pickle is None) and args.save_loaders:
        with open("pickled_training_loader.pickle", "wb") as handle:
            pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("pickled_validating_loader.pickle", "wb") as handle:
            pickle.dump(validation_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(">>> The training and validating loaders have been pickled.\n")

    if args.verbose:
        print(">>> Split original dataset into train/validate datasets")
        print(f">>> Training dataset was built from a dataset of {len( train_loader.dataset )} instances")
        print(f"... and made into {len( train_loader )} minibatches of {train_loader.batch_size} instances")
        print(
            f"... ( {len( train_loader )} * {train_loader.batch_size} = {len( train_loader ) * train_loader.batch_size} )\n")
        print(f">>> Validating dataset was built from a dataset of {len( validation_loader.dataset )} instances")
        print(f"... and made into {len( validation_loader )} minibatches of {validation_loader.batch_size} instances")
        print(
            f"... ( {len( validation_loader)} * {validation_loader.batch_size} = {len( validation_loader ) * validation_loader.batch_size} )\n")

        # making sure the datasets are balanced wrt labels
        training_labels, validating_labels = [], []
        with torch.no_grad():
            for _, target in train_loader:
                training_labels.extend([t.item() for t in target])
            for _, target in validation_loader:
                validating_labels.extend([t.item() for t in target])

        training_labels_counter = Counter(training_labels)
        for k, v in training_labels_counter.items():
            training_labels_counter[k] = '{:.3f}%'.format(100. * v / len(training_labels))

        validating_labels_counter = Counter(validating_labels)
        for k, v in validating_labels_counter.items():
            validating_labels_counter[k] = '{:.3f}%'.format(100. * v / len(validating_labels))

        del training_labels  # for the sake of memory management
        del validating_labels  # for the sake of memory management
        print(">>> The label distribution in the training dataset:\n{}\n".format(
            '\n'.join([(k, v).__str__() for k, v in training_labels_counter.items()])))
        print(">>> The label distribution in the validating dataset:\n{}\n".format(
            '\n'.join([(k, v).__str__() for k, v in validating_labels_counter.items()])))

    try:
        assert (len(train_loader.dataset) * (1.0 - args.validation_split_fraction)) % train_loader.batch_size == 0
        assert (len(validation_loader.dataset) * args.validation_split_fraction) % validation_loader.batch_size == 0
    except AssertionError as err:
        raise Exception(
            f"Error: your chosen test/val split ({1.0-args.validation_split_fraction} , {args.validation_split_fraction}) doesn't match your minibatch size ({args.batch_size}).\nMake sure that both your train and validation datasets' length is a multiple of your args.batch-size argument.") from err

    start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    logpath = os.path.join(os.getcwd(), 'logs', start_timestamp)

    if not os.path.isdir(logpath):
        os.mkdir(logpath)

    # logger = Logger(logpath)
    # if args.n_models > 1:
    #     loggers = [Logger(
    #         os.path.join(os.getcwd(), 'logs', 'MNIST-Sanity-Check' + start_timestamp + f'{i}_of_{args.n_models}')) for i
    #                in range(args.n_models)]

    if args.verbose:
        print(f"\nThe log file will be saved in {logpath.__str__()}\n")

    # Model definition
    # best performing model:
    model = ThreeLayerModel13().to(
        device).double()  # casting it to double because of some pytorch expected type peculiarities


    # Loss and optimizer
    # optimizer = torch.optim.Adam( model.parameters(), lr=args.lr )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # adding l2 loss

    loss_fn = nn.MSELoss()

    print("\n>>> Starting training\n")

    # dummy declarations, get overwritten at the final epoch to contain
    # all_models_final_outputs: a 40,000 x ( 10 * # models ) feature numpy array
    # all_corresponding_targets: a 40,000 target numpy vector containing the label for each row in all_models_final_outputs
    all_models_final_outputs, all_corresponding_targets = None, None

    if args.MNIST_sanity_check == True:
        for epoch in range(args.epochs):
            sanity_check_train(args, model, device, train_loader, optimizer, epoch,
                               (1.0 - args.validation_split_fraction), logger)
            sanity_check_validate(args, model, device, validation_loader, epoch, logger, args.validation_split_fraction)

    else:
        if args.n_models == 1:

            for epoch in range(args.epochs):
                training_output, training_targets = train(args, model, loss_fn, device, train_loader, validation_loader,
                                                          optimizer, epoch, args.batch_size, logger)
                validating_output, validating_targets = validate(args, model, loss_fn, device, validation_loader, epoch,
                                                                 logger, args.validation_split_fraction)

                if epoch == (args.epochs - 1):  # we only care about the last epoch's output
                    all_models_final_outputs = np.vstack((training_output, validating_output))
                    all_corresponding_targets = np.hstack(
                        (training_targets, validating_targets))  # need to hstack vectors

        else:
            print("This doesn't work... Exiting now...")
            raise SystemExit

            for model_iteration in range(args.n_models):
                this_train_loader, this_validation_loader = deepcopy(train_loader), deepcopy(validation_loader)
                this_model = Elliot_Model().to(device).double()
                print(f">>> training model {model_iteration+1} of {args.n_models}")

                for epoch in range(args.epochs):
                    training_output, training_targets = train(args, this_model, loss_fn, device, this_train_loader,
                                                              this_validation_loader, optimizer, epoch, args.batch_size,
                                                              loggers[model_iteration])
                    validating_output, validating_targets = validate(args, this_model, loss_fn, device,
                                                                     this_validation_loader, epoch,
                                                                     loggers[model_iteration],
                                                                     args.validation_split_fraction)

                    if epoch == (args.epochs - 1):  # we only care about the last epoch's output
                        if model_iteration == 0:
                            all_models_final_outputs = np.vstack((training_output, validating_output))
                            all_corresponding_targets = np.hstack(
                                (training_targets, validating_targets))  # need to hstack vectors

                        else:
                            all_models_final_outputs = np.hstack((all_models_final_outputs, np.vstack(
                                (training_output, validating_output))))  # hstack is not a typo

    # Saving output
    if (args.save_model):

        if args.n_models > 1:

            for e, this_model in enumerate(models):
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'pickled-params',
                                                            start_timestamp + f"_model_{e+1}_of_{args.n_models}.savefile"))

            if args.merge_train_validate_outputs:
                pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                           start_timestamp + f"_{args.n_models}_models_merged_training_and_validating_outputs_and_targets.pickle")
                with open(pickle_path, 'wb') as meta_clf_feature_mat:
                    pickle.dump(np.hstack((all_models_final_outputs, all_corresponding_targets.reshape(-1, 1))),
                                meta_clf_feature_mat, protocol=pickle.HIGHEST_PROTOCOL)

                print(
                    f">>> The <features from training & validating><labels from training & validating> matrix pickle for meta-classification is saved under\n{pickle_path}")

            else:
                training_and_validating_outputs = np.hstack(
                    (all_models_final_outputs, all_corresponding_targets.reshape(-1, 1)))

                training_indices = range(0, int(len(train_loader.dataset) * (1.0 - args.validation_split_fraction)))
                validating_indices = range(max(training_indices) + 1, len(train_loader.dataset))

                for m in range(args.n_models):
                    column_range = list(range(m * 10, (m + 1) * 10))
                    column_range.append(training_and_validating_outputs.shape[1] - 1)
                    output_subarray_from_training = training_and_validating_outputs[training_indices, :]
                    output_subarray_from_training = output_subarray_from_training[:, column_range]

                    output_subarray_from_validating = training_and_validating_outputs[validating_indices, :]
                    output_subarray_from_validating = output_subarray_from_validating[:, column_range]

                    pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                               start_timestamp + f"_model_{m+1}_training_outputs_and_targets.pickle")
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(output_subarray_from_training, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    print(
                        f">>> The <features from training><labels from training> matrix pickle for meta-classification is saved under\n{pickle_path}")

                    pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                               start_timestamp + f"_model_{m+1}_validating_outputs_and_targets.pickle")
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(output_subarray_from_validating, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    print(
                        f">>> The <features from validating><labels from validating> matrix pickle for meta-classification is saved under\n{pickle_path}")

        else:
            torch.save(model.state_dict(),
                       os.path.join(os.getcwd(), 'pickled-params', start_timestamp + '_model.savefile'))

            if args.merge_train_validate_outputs:
                pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                           start_timestamp + "_merged_training_and_validating_outputs_and_targets.pickle")

                with open(pickle_path, 'wb') as meta_clf_feature_mat:
                    pickle.dump(np.hstack((all_models_final_outputs, all_corresponding_targets.reshape(-1, 1))),
                                meta_clf_feature_mat, protocol=pickle.HIGHEST_PROTOCOL)

                print(
                    f">>> The <features from training & validating><labels from training & validating> matrix pickle for meta-classification is saved under\n{pickle_path}")

            else:
                training_and_validating_outputs = np.hstack(
                    (all_models_final_outputs, all_corresponding_targets.reshape(-1, 1)))

                training_indices = range(0, int(len(train_loader.dataset) * (1.0 - args.validation_split_fraction)))
                validating_indices = range(max(training_indices) + 1, len(train_loader.dataset))

                for m in range(args.n_models):
                    column_range = list(range(m * 10, (m + 1) * 10))
                    column_range.append(training_and_validating_outputs.shape[1] - 1)
                    output_subarray_from_training = training_and_validating_outputs[training_indices, :]
                    output_subarray_from_training = output_subarray_from_training[:, column_range]

                    output_subarray_from_validating = training_and_validating_outputs[validating_indices, :]
                    output_subarray_from_validating = output_subarray_from_validating[:, column_range]

                    pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                               start_timestamp + f"_training_outputs_and_targets.pickle")
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(output_subarray_from_training, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    print(
                        f">>> The <features from training><labels from training> matrix pickle for meta-classification is saved under\n{pickle_path}")

                    pickle_path = os.path.join(os.getcwd(), 'pickled-params',
                                               start_timestamp + f"_validating_outputs_and_targets.pickle")
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(output_subarray_from_validating, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    print(
                        f">>> The <features from validating><labels from validating> matrix pickle for meta-classification is saved under\n{pickle_path}")

        with open(os.path.join(os.getcwd(), 'pickled-params', start_timestamp + '_params.savefile'),
                  'w') as params_file:
            params_file.write(args.__repr__())
            params_file.write('\n')
            params_file.write(optimizer.__repr__())
            params_file.write('\n')
            params_file.write(loss_fn.__repr__())

    print(f"\nThe log file was saved in {logpath.__str__()}\n")
    print(f"\nThe model and parameter save files were saved in { os.path.join( os.getcwd(), 'pickled-params' ) }\n")