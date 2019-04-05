import pickle, os

import numpy as np 
import pandas as pds 

from scipy.sparse import csr_matrix

def triangularize_matrix( mat ):
    """
    Squarifies a triangular matrix/array.
    """
    return np.tril( mat ) + np.tril( mat, -1 ).T

def create_matrix( path_to_downsampled_file, read_csv_params, output_type:str, output_filename:str,  resolution=10000, as_pickle=False, enforce_symm=False ):
    
    output_type = output_type.lower()
    assert output_type in [ "numpy", "pandas", "sparse" ]
    assert os.path.isfile( path_to_downsampled_file )
    downsampled_df = pds.read_csv( path_to_downsampled_file, *read_csv_params )

    matrix_dims = 1 + max( downsampled_df[0] ) / resolution # the + 1 is to account for the 0-<res> bin

    hic_mat = np.zeros( ( matrix_dims, matrix_dims ) )
    downsampled_df_concise = downsampled_df.groupby( [0,1] ).sum()
    for row in downsampled_df_concise.iterrows():
        if row[0] <= row[1]:
            hic_mat[ row[0], row[1] ] += row[2]
        else:
            hic_mat[ row[1], row[0] ] += row[2]
    
    if enforce_symm:
        upper_triangular_sum = np.sum( np.triu( hic_mat ) )
        lower_triangular_sum = np.sum( np.tril( hic_mat ) )
        assert ( lower_triangular_sum == 0.0 ) or ( upper_triangular_sum == 0.0 ) # ensure the previous for-loop built a triangular matrix
        if upper_triangular_sum != lower_triangular_sum:
            hic_mat = triangularize_matrix( hic_mat )

    if output_type == "pandas":
        output = pds.DataFrame( hic_mat, columns=[], index=[]) # fix this
        if as_pickle:
            with open( output_filename, 'wb' ) as pickle_handle:
                pickle.dump( output, handle, protocol=pickle.HIGHEST_PROTOCOL )
        else:
            output.to_csv( output_filename, sep='\t', header=output.columns, index=output.index ) # fix float_format argument

    elif output_type == "sparse":
        output = csr_matrix( hic_mat )
        with open( output_filename, 'wb' ) as pickle_handle:
            pickle.dump( output, handle, protocol=pickle.HIGHEST_PROTOCOL )

    else: # output_type == "numpy"
        if as_pickle:
            with open( output_filename, 'wb' ) as pickle_handle:
                pickle.dump( output, handle, protocol=pickle.HIGHEST_PROTOCOL )
        else:
            np.savetxt( output_filename, output, delimiter='\t',  ) # fix fmt argument
