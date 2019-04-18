import os, argparse, pickle
import numpy as np
from utils.unpickle import gather_chromosome_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def train_and_test(X_train, y_train, X_test, y_test, args):
    result_mse = list()
    result_error = list()
    for _ in range(args.num_iterations):
        model = KNeighborsRegressor(
            n_neighbors=args.k_neighbors,
            weights='distance'
        )
        print( "Fitting model...\n" ) 
        model.fit(X_train, y_train)
        print( "Fit model, making predictions...\n" )
        y_hat = model.predict(X_test)
        print( "Made predictions, calculating MSE & RMSE...\n" )
        error = mean_squared_error(y_test, y_hat)
        result_mse.append(error)
        result_error.append(np.sqrt(error))
        if args.verbose:
            print('MSE on test set: ' + str(error))
            print('Mean error on test set: ' + str(np.sqrt(error)))
            print('------------------------')
    return result_mse, result_error


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Used to run K-Nearest-Neighbors Regressor')
    parser.add_argument('--train-data-path', type=str, required=True, metavar='T',
                        help="path to the dir containing the pickled list of (array, target, chromosome number) tuples for training")
    parser.add_argument('--valid-data-path', type=str, required=True, metavar='V',
                        help="path to the dir containing the pickled list of (array, target, chromosome number) tuples for testing")
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--num-iterations', type=int, default=1, metavar='I',
                        help='num iterations (default: 1)')
    parser.add_argument('--k-neighbors', type=int, default=5, metavar='K',
                        help='num iterations (default: 1)')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='boolean indicator of verbosity')
    parser.add_argument("--data-transform", type=str, default="None",
                        help="transform to apply to the data before passing to the CNNs (can be 'mult_cap' or 'log').")

    args = parser.parse_args()

    print('Loading data...')

    # X, y, _ = gather_chromosome_data( args.valid_data_path )
    X_test, y_test, _ = gather_chromosome_data(args.valid_data_path)
    X_train, y_train, _ = gather_chromosome_data(args.train_data_path)

    if args.verbose:
        print('Data loaded')

    # # small sample for now to debug, needs to be replaced with real full train / test data
    # X_train = X[:10000]
    # y_train = y[:10000]
    
    # X_test = X[10000:20000]
    # y_test = y[10000:20000]

    if args.data_transform == "mult_cap":
        # scale by downsample ratio
        print( "\n>>> Multiplying arrays (not targets) by 16.0 and capping arrays and targets at 100.0\n" )
        X_train = [ np.clip( arr*16.0, 0.0, 100.0 ) for arr in X_train ]
        X_test = [ np.clip( arr*16.0, 0.0, 100.0 ) for arr in X_test ]
        y_train = [ max( min( target, 100.0 ), 0.0 ) for target in y_train ] 
        y_test = [ max( min( target, 100.0 ), 0.0 ) for target in y_test ]  
    elif args.data_transform == "log":
        print( "\n>>> Appling log( arr + 1.0 ) transform\n" )
        X_train = [ np.log( arr+1.0 ) for arr in X_train ]
        X_test = [ np.log( arr+1.0 ) for arr in X_test ]
        y_test = np.log( np.array( y_test ) + 1.0 )
        y_train = np.log( np.array( y_train ) + 1.0 )

    print( f"{len( X_train )}, {len( X_test )}\n" )
    X_test = np.array([x.flatten() for x in X_test])
    X_train = np.array([x.flatten() for x in X_train])

    mses, errors = train_and_test(X_train, y_train, X_test, y_test, args)

    print('All errors mean:' + str(np.mean(errors)))
    print('All errors std-dev: ' + str(np.std(errors)))
    print('done')