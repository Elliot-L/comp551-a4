import numpy as np 
from scipy.ndimage import convolve 

def make_window_avgd_matrix( arr:np.ndarray, window_size:int, convolve_args={'mode':'nearest'}, ret_type=None ):
    """
    Wrapper around vectorized window-average function (see scipy.ndimage.convolve).

    Arguments:

        arr: np.ndarray for which we want to get the window averages.

        window_size: int specifying the size of the square window (should be an odd integer).

        convolve_args:  additional parameters to pass onto scipy.ndimage.convolve 
                        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html)
                    
        ret_type: optional specifier for the window-averaged array's dtype (e.g. int or float).

    Returns:

        avgd_arr: np.ndarray copy of the input arr, but after it has been averaged using the specified window size.
    """
    # making averaging kernel
    kernel = np.ones( ( window_size, window_size ) ) / ( window_size**2 )
    assert np.sum( kernel ) == 1.0

    avgd_arr = convolve( arr, kernel, **convolve_args )

    if ret_type is not None:
        return avgd_arr.astype( ret_type )
    else:
        return avgd_arr
