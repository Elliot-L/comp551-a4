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

def iterate_over_diagonals( arr:np.ndarray, bandwidth:int, window_dim:int ):
    """
    Returns a list of (window_array centered at some point x,y on a diagonal (a np.ndarray), diagonal offset from main diagonal (an int)) tuples.

    Arguments:

        arr: original np.ndarray to iterate over.

        bandwidth: int representing how many diagonals to iterate over.

        window_dim: odd integer representing the size of the window arrays.
    
    Returns:

        centerpoints: list of tuples of two elements:
                        first element: a window_dim x window_dim subarray centered at some point along some diagonal.
                        second element: the indices of the centerpoint of first element.

    """
    window_pad = ( window_dim - 1 ) // 2
    centerpoints = {}
    for diag_offset in range( bandwidth ):
        relevant_indices = kth_diag_indices( 
            arr[ window_pad:-window_pad, window_pad:-window_pad ], 
            diag_offset 
        )
        centerpoints[ diag_offset ] = relevant_indices
    
    return centerpoints

def get_diagonal_wise_windows( arr, num_diags, window_dim ):

    diag_centerpoint_inds = iterate_over_diagonals( arr, num_diags, window_dim )
    diag_wise_windows = {}
    for diag_offset, centerpoint_inds in diag_centerpoint_inds.items():
        diag_wise_windows[ diag_offset ] = [ ]
        for ( centerpoint_row_i, centerpoint_col_i ) in tqdm( zip( centerpoint_inds ) ):
            diag_wise_windows[ diag_offset ].append( 
                arr[ 
                    centerpoint_row_i-window_dim:centerpoint_row_i+window_dim+1, 
                    centerpoint_col_i-window_dim:centerpoint_col_i+window_dim+1 
                ], 
                ( centerpoint_row_i, centerpoint_col_i )
            )
    
    return diag_wise_windows

        
