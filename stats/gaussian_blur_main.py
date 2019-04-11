import cv2
import numpy as np 
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal
import scipy.stats as st
import scipy

from mpl_toolkits import mplot3d

from skimage.filters import gaussian

def my_make_gaussian_kernel( dim, mu: list, cov_mat: np.ndarray, xrange=(-5,5), showplot=False, normalize=False ):
    """
    Function used to define some multivariate gaussian kernels (and plot them).
    Ref: https://stackoverflow.com/questions/48465683/visualizing-a-multivariate-normal-distribution-with-numpy-and-matplotlib-in-3-di/48466089
    """
    x = y = np.linspace( *xrange, dim ) 
    X, Y = np.meshgrid( x, y )
    pos = np.dstack( ( X, Y) )
    rv = multivariate_normal( mu, cov_mat )
    Z = rv.pdf( pos )
    if normalize: 
        Z /= np.sum( Z )
    if showplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        plt.show()
    return Z

def gaussian_blur( in_array, gaussian_blur_function, gaussian_blur_parameters={}, verbose=False ):
    """
    Main function to perform a gaussian blur.

    Arguments:

        in_array: the np.ndarray (or scipy.sparse array) representing the matrix to blur.

        gaussian_blur_function: one of "scipy", "skimage", or "cv2"; indicates which gaussian blur function to use.

        gaussian_blur_parameters:   parameters to pass onto the specified gaussian blur function. 
                                    if "scipy" or "skimage", the parameters (and default values) are:
                                        image, sigma=1, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0
                                        where image, sigma, and mode are the key parameters.
                                        links to documentation: 
                                        http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
                                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
                                    
                                    if "cv2", the parameters (and default values) are:
                                        src, ksize, sigmaX, sigmaY, borderType
                                        where src, ksize, and sigmaX are the key parameters
                                        e.g. cv2.GaussianBlur( some_array, (5,5), sigmaX=3 )
                                        link to documentation: 
                                        https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
        
        verbose: boolean indicator of verbosity
    
    Returns:

        The blurred version of the input array (with the same type).

    """
    return_type = type( in_array )
    gaussian_blur_function = gaussian_blur_function.lower()
    try:
        assert gaussian_blur_function in [ "scipy", "skimage", "cv2" ]
    except AssertionError as aerr:
        raise AssertionError( f"\ngaussian_blur_function must be 'scipy', 'skimage', or 'cv2'. You passed {gaussian_blur_function}.\n" ) from aerr

    if isinstance( in_array, csr_matrix ) and in_array.dtype == float:
        if verbose:
            print( "\nInput to gaussian_blur is a sparse matrix, inflating, blurring, and re-sparsifying\n" )
        in_array = in_array.toarray()

    # skimage.filters.gaussian is a wrapper around scipy.ndimage.filters.gaussian_filter
    if ( gaussian_blur_function == "scipy" ) or ( gaussian_blur_function == "skimage" ): 
        if return_type == csr_matrix:

            return csr_matrix( gaussian_filter( in_array.toarray().astype( float ), **gaussian_blur_parameters ) ) 
        else:
            return gaussian_filter( in_array.astype( float ), **gaussian_blur_parameters )
    else:
        if return_type == csr_matrix:
            return csr_matrix( cv2.GaussianBlur( in_array.toarray().astype( float ), **gaussian_blur_parameters ) )
        else:
            return cv2.GaussianBlur( in_array.toarray().astype( float ), **gaussian_blur_parameters )

# copied from https://github.com/zhangyan32/HiCPlus/blob/master/src/Gaussian_kernel_smoothing.py
def gkern(kernlen, nsig): # holy fucking shit, he literally copied the *_wrong_* original answer from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel   

def Gaussian_filter(matrix, sigma=4, size=13):
    result = np.zeros(matrix.shape)
    padding = size / 2 # causes a bug with floats
    kernel = gkern(13, nsig=sigma)
    for i in range(padding, matrix.shape[0] - padding):
        for j in range(padding, matrix.shape[0] - padding):
            result[i][j] = np.sum(matrix[i - padding : i + padding + 1, j - padding : j + padding + 1] * kernel)
    return result

def compare_gaussian_kernels( dim, sigma, my_make_gaussian_kernel_params={'showplot':'True','normalize':'True'} ):

    their_gbk = gkern( dim, sigma )
    my_gbk = my_make_gaussian_kernel( dim, [ 0.0, 0.0 ], np.eye(2)*sigma, **my_make_gaussian_kernel_params )
    
    print( np.sum( their_gbk ) ) 
    print( np.sum( my_gbk ) )

    fig = plt.figure() 
    ax = fig.subplots( nrows=1, ncols=3 )
    ax[0].imshow( their_gbk, cmap="Reds" )
    ax[1].imshow( my_gbk, cmap="Reds" )
    ax[2].imshow( my_gbk - their_gbk )
    plt.show()



if __name__ == "__main__":
    #mine = my_make_gaussian_kernel( 13, mu=[.0,.0], cov_mat=np.eye(2)*4., showplot=True, normalize=True )   
    compare_gaussian_kernels( 13, 4.0 )