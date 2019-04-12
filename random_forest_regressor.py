import os, argparse, pickle
import numpy as np
from utils.unpickle import unpickle_data_pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Used to run RandomForestRegressor')
    parser.add_argument('--data-pickle-path', type=str, required=True, metavar='P',
                        help="path to the file containing the pickled list of (array, target, chromosome number) tuples")
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='boolean indicator of verbosity')

    args = parser.parse_args()

    print('Loading data...')

    # todo: need the proper logic for loading all training and validation data appropriately
    X, y, data_chromnum_list = unpickle_data_pickle( args.data_pickle_path )

    if args.verbose:
        print('Data loaded')

    model = RandomForestRegressor(random_state=args.seed)

    # small sample for now to debug, needs to be replaced with real full train / test data
    X_train = X[:100]
    y_train = y[:100]

    X_test = X[100:200]
    y_test = y[100:200]

    X_train = np.array([x.flatten() for x in X_train])
    X_test = np.array([x.flatten() for x in X_test])

    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    error = mean_squared_error(y_test, y_hat)
    print('MSE on test set: ' + str(error))
    print('Mean error on test set: ' + str(np.sqrt(error)))


    print('done')