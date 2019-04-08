import numpy as np 

from scipy.stats import pearsonr, spearmanr 

def score_correlation( arr1:np.ndarray, arr2:np.ndarray, triangular="upper", metric="pearson" ):
    """
    Wrapper around scipy.stats.pearsonr and scipy.stats.spearmanr to quantify the correlation between two matrices' values.

    Arguments:

        arr1: symmetric or triangular np.ndarray of size M x M. 

        arr2: symmetric or triangular np.ndarray also of size M x M.

        triangular: str indicating whether to compare the upper/lower triangular of the two input arrays. 

        metric: str indicating whether to use the spearman correlation coefficient or pearson correlation coefficient.

    Returns:

        The spearman or pearson correlation coefficient of the values in the two input arrays. 
    """
    # argument validation 
    triangular = triangular.lower()
    try:
        assert triangular in [ "upper", "lower", "already flattened" ]
    except AssertionError as ae:
        raise AssertionError( f"\ntriangular can be one of 'upper', 'lower', or 'already flattened'; you passed: '{triangular}'\n") from ae
    
    metric = metric.lower()
    try:
        assert metric in [ "pearson", "spearman" ]
    except AssertionError as ae:
        raise AssertionError( f"\nmetric can be one of 'pearson' or 'spearman'; you passed: '{metric}'\n") from ae

    inds = None
    if triangular == "already flattened":
        
        if metric == "pearson":
            return pearsonr( flat_arr1, flat_arr2 )
        else: # metric == "spearman"
            return spearmanr( flat_arr1, flat_arr2 )

    elif triangular == "upper":
        inds = np.triu_indices( arr1.shape[0] )
        
    else: # triangular == "lower"
        inds = np.tril_indices( arr1.shape[0] )
    
    flat_arr1 = arr1[ inds ]
    flat_arr2 = arr2[ inds ]

    if metric == "pearson":
        return pearsonr( flat_arr1, flat_arr2 )
    else: # metric == "spearman"
        return spearmanr( flat_arr1, flat_arr2 )