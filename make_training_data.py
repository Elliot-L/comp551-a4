import os, pickle, argparse

import numpy as np 
from tqdm import tqdm

from utils.sparse_matrix_construction import *
from utils.iterate_over_diagonal import *

def make_training_data_for_downsampled_dir( raw_dir, downsampled_dir, bandwidth:int, window_dim:int, pickle_file_path_prefix="pickled_data_for_" ):
    """
    Wrapper around make_training_data_list to loop it over pairs of files from two directories.

    Arguments:

        raw_dir: str or path to the directory containing the raw data .gz files.

        downsampled_dir: str of path to the directory containing the downsampled data .gz files.

        bandwidth: int for make_training_data_list specifying how many diagonals to iterate over.

        window_dim: odd int for make_training_data_list indicating the dimension of the window/subarray.

        pickle_file_path_prefix: optional prefix to prepend to the output pickle files.

    Returns:
        
        nothing
    """
    for d in [ raw_dir, downsampled_dir ]:
        assert os.path.isdir( d )
    
    raw_files = [ os.path.join( raw_dir, f ) for f in os.listdir( raw_dir ) if '.gz' in f ]
    downsampled_files = [ os.path.join( downsampled_dir, f ) for f in os.listdir( downsampled_dir ) if '.gz' in f ]

    assert len( raw_files ) == len( downsampled_files )
    raw_files.sort( key=lambda filepath: os.path.basename( filepath ).split('_')[0] )
    downsampled_files.sort( key=lambda filepath: os.path.basename( filepath ).split('_')[0] )

    for raw_file, downsampled_file in list( zip( raw_files, downsampled_files ) )[::-1]:
        print( f"\nstarting for chromosome {os.path.basename( raw_file ).split( '_' )[0]}\n" )
        make_training_data_list(
            
            create_matrix( # raw_array
                raw_file,
                {
                    'sep':'\t',
                    'header':None,
                    'index_col':None,
                    'compression':'gzip'
                },
                "numpy"
            ),
            create_matrix( # downsampled_array 
                downsampled_file,
                {
                    'sep':'\t',
                    'header':None,
                    'index_col':None,
                    'compression':'gzip'
                },
                "numpy"
            ),
            bandwidth,
            window_dim,
            os.path.basename( raw_file ).split( '_' )[0], # chrnum
            os.path.join( os.path.dirname( downsampled_file ), f"{pickle_file_path_prefix}{os.path.basename( downsampled_file )[:-3]}.pickle" )
        )

def make_training_data_list( raw_arr:np.ndarray, ds_arr:np.ndarray, bandwidth:int, window_dim:int, chrnum:int, pickle_file_path=None ):
    """
    Function to iterate over raw_arr and ds_arr's 0-bandwidth's diagonals and extract a list of tuples of
    ( window_dim x window_dim subarray of ds_arr centered along one of the diagonals, the corresponding subarray's centerpoint in raw_arr, and chrnum ).

    Arguments:

        raw_arr: np.ndarray corresponding to the reference array (the one used to select the second element of each tuple; the "target value").

        ds_arr: np.ndarray corresponding to the array which we want to segment into subarrays.

        bandwidth: int indicating how many diagonals to iterate over.

        window_dim: odd integer indicating the dimension of the window/subarray.

        chrnum: int specifying which chromosome these arrays were obtained from.

        pickle_file_path: optional string/path specifying where to save the list of tuples.

    Returns:

        data_list:  list of tuples like the following:
                    ( window_dim x window_dim subarray of ds_arr centered along one of the diagonals, 
                    the corresponding subarray's centerpoint in raw_arr,
                    chrnum )
    """
    # ensuring the matrices are of the same dimensions
    if raw_arr.shape != ds_arr.shape:
        largest_dim = max( raw_arr.shape[0], ds_arr.shape[0] )
        if ds_arr.shape[0] != largest_dim:
            ds_arr = np.pad( ds_arr, ( ( 0, largest_dim-ds_arr.shape[0] ), ( 0, largest_dim-ds_arr.shape[0] ) ), 'constant', constant_values=0 )
        else: # ds_arr is larger than raw_arr (unlikely)
            raw_arr = np.pad( raw_arr, ( ( 0, largest_dim-raw_arr.shape[0] ), ( 0, largest_dim-raw_arr.shape[0] ) ), 'constant', constant_values=0 )
    assert raw_arr.shape == ds_arr.shape 
    
    window_pad = ( window_dim - 1 ) // 2
    data_list = []    
    for diag_offset in tqdm( range( bandwidth ) ):

        # rdci is short for relevant_diagonal_centerpoint_indices
        rdci_rows, rdci_cols = kth_diag_indices( raw_arr, diag_offset ) # [ window_pad:-window_pad ]

        for rdci_row, rdci_col in zip( rdci_rows[ window_pad:-window_pad ], rdci_cols[ window_pad:-window_pad ] ): 
            data_list.append( 
                ( 
                    ds_arr[ rdci_row-window_pad:rdci_row+window_pad+1, rdci_col-window_pad:rdci_col+window_pad+1 ], 
                    raw_arr[ rdci_row, rdci_col ], 
                    chrnum 
                ) 
            )

    if pickle_file_path is not None:
        with open( pickle_file_path, 'wb' ) as pickle_handle:
            pickle.dump( data_list, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )
        print( f"\nsaved output in\n{pickle_file_path}\n")

    return data_list

if __name__ == '__main__':
    """
    # example for debugging
    arr = np.random.randint( 0, 10, (10,10) )
    arr[ np.tril_indices( arr.shape[0] ) ] = 0
    make_training_data_list( arr, arr, 3, 5, 42, pickle_file_path="test_pickle_file_path.pickle" )
    """

    parser = argparse.ArgumentParser( description='Making data for CNNs' )
    parser.add_argument( '--raw-data-dir', type=str, metavar='R', required=True,
                        help='path to the directory containing the raw data files.' )
    parser.add_argument( '--downsampled-data-dir', type=str, metavar='D', required=True,
                        help='path to the directory containing the downsampled data files.' )
    parser.add_argument( '--bandwidth', type=int, metavar='B', default=100,
                        help='how many diagonals to iterate over (default=100).' )
    parser.add_argument( '--window-dim', type=int, metavar='W', default=13,
                        help='odd int representing window size (default=13).' )
    parser.add_argument( '--pickle-prefix', type=str, metavar='P', default="pickled_data_for_",
                        help='prefix for pickled files (default="pickled_data_for_").' )
    args = parser.parse_args()

    try:
        assert args.window_dim % 2 == 1 
    except AssertionError as ae:
        raise AssertionError( f"\nthe window-dim argument must be an odd integer; you passed {args.window_dim}\n" ) from ae 
    
    make_training_data_for_downsampled_dir(
        args.raw_data_dir, 
        args.downsampled_data_dir, 
        args.bandwidth,
        args.window_dim,
        f"{args.pickle_prefix}_({args.bandwidth}_{args.window_dim})_"
    )
