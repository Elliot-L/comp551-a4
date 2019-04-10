import os, pickle

import numpy as np 

from tqdm import tqdm 

def kth_diag_indices( a:np.ndarray, k:int ):
    """
    Returns the indices of the k-th diagonal of the array a.

    Arguments:

        a: the numpy array of interest.

        k: integer representing the diagonal offset ( ranges from [ (-1 * #rows-1), (#rows-1) ] ).

    Returns:
        
        rows: numpy vector of the kth diagonal's row indices.

        cols: numpy vector of the kth diagonal's column indices.

    """
    rows, cols = np.diag_indices_from( a )
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def make_training_data( raw_arr:np.ndarray, ds_arr:np.ndarray, bandwidth:int, window_dim:int, chrnum:int, pickle_file_path=None ):

    # ensuring the matrices are of the same dimensions
    if raw_arr.shape != ds_arr.shape:
        largest_dim = max( raw_arr.shape[0], ds_arr.shape[0] )
        if ds_arr.shape[0] != largest_dim:
            ds_arr = np.pad( ds_arr, ( ( 0, largest_dim-ds_arr.shape[0] ), ( 0, largest_dim-ds_arr.shape[0] ) ), 'constant', constant_values=0 )
        else: # ds_arr is larger than raw_arr (unlikely)
            raw_arr = np.pad( raw_mat, ( ( 0, largest_dim-raw_arr.shape[0] ), ( 0, largest_dim-raw_arr.shape[0] ) ), 'constant', constant_values=0 )
    assert raw_arr.shape == ds_arr.shape 
    
    window_pad = ( window_dim - 1 ) // 2
    data_list = []    
    for diag_offset in range( bandwidth ):

        # rdci is short for relevant_diagonal_centerpoint_indices
        rdci_rows, rdci_cols = kth_diag_indices( raw_arr, diag_offset ) # [ window_pad:-window_pad ]
        print( rdci_rows, rdci_cols )

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

def iterate_over_diagonals( arr:np.ndarray, bandwidth:int, window_dim:int ):
    """
    Returns a list of (window_array centered at some point x,y on a diagonal (a np.ndarray), diagonal offset from main diagonal (an int)) tuples.

    Arguments:

        arr: original np.ndarray to iterate over.

        bandwidth: int representing how many diagonals to iterate over.

        window_dim: odd integer representing the size of the window arrays.
    
    Returns:

        windows_and_centerpoints: dictionary of diagonal_offset:tup key:value pairs, where tup is a tuple of two elements:
                        first element: a window_dim x window_dim subarray centered at some point along some diagonal.
                        second element: the indices of the centerpoint of first element.

    """
    window_pad = ( window_dim - 1 ) // 2
    windows_and_centerpoints = {}
    for diag_offset in range( bandwidth ):
        relevant_indices = kth_diag_indices( 
            arr[ window_pad:-window_pad, window_pad:-window_pad ], 
            diag_offset 
        )
        windows_and_centerpoints[ diag_offset ] = relevant_indices
    
    return windows_and_centerpoints

def get_diagonal_wise_windows( arr:np.ndarray, num_diags:int, window_dim:int ):
    """
    Wrapper for iterate_over_diagonals that groups together its outputs in a dictionary with
    diagonal_offset::( window_array, ( centerpoint_row_index, centerpoint_column_index ) ) key::value pairs.

    Arguments:

        arr: upper triangular (or symmetric) np.ndarray used in scanning window process.

        num_diags: int representing how many diagonals to iterate over.

        window_dim: odd integer representing the size of the window arrays.

    Returns:

        diag_wise_windows: dictionary of diagonal_offset::( window_array, ( centerpoint_row_index, centerpoint_column_index ) ) key::value pairs.

    """
    diag_centerpoint_inds = iterate_over_diagonals( arr, num_diags, window_dim )
    diag_wise_windows = {}
    for diag_offset, centerpoint_inds in diag_centerpoint_inds.items():
        diag_wise_windows[ diag_offset ] = [ ]
        for ( centerpoint_row_i, centerpoint_col_i ) in tqdm( zip( centerpoint_inds ) ):
            diag_wise_windows[ diag_offset ].append( 
                (
                    arr[ 
                        centerpoint_row_i-window_dim:centerpoint_row_i+window_dim+1, 
                        centerpoint_col_i-window_dim:centerpoint_col_i+window_dim+1 
                    ], 
                    ( centerpoint_row_i, centerpoint_col_i )
                )
            )
    
    return diag_wise_windows

def make_mat_with_only_k_diags( upper_triangular_arr:np.ndarray, k=200 ):
    """
    Makes a version of input array which only contains its entries in the 0 to kth diagonals.

    Arguments:

        upper_triangular_arr: upper triangular (or symmetric) np.ndarray whose k diagonal entries we want to copy.

        k: int indicating how many diagonals to iterate over.

    Returns:

        ret_arr:    np.ndarray of the same shape as the input array but whose only non-zero values are
                    those in the 0-kth diagonals of the input array.
                    
    """
    ret_arr = np.zeros( upper_triangular_arr.shape )
    for diag in range( k ):
        diag_inds = kth_diag_indices( upper_triangular_arr, diag )
        ret_arr[ diag_inds ] += upper_triangular_arr[ diag_inds ]
    return ret_arr 