import pickle, os

import numpy as np 
import pandas as pds 

from scipy.sparse import csr_matrix

def triangularize_matrix( ut_mat ):
    """
    Squarifies an upper triangular matrix/array.
    """
    return np.triu( ut_mat ) + np.triu( ut_mat, 1 ).T

def create_matrix( path_to_downsampled_file, read_csv_params, output_type:str, output_filename:str, resolution=10000, as_pickle=False, enforce_symm=False ):
    
    # input validation
    output_type = output_type.lower()
    try:
        assert output_type in [ "numpy", "pandas", "sparse" ]
    except AssertionError as ae:
        raise AssertionError( f"\noutput_type can be one of 'numpy', 'pandas', 'sparse'; you passed: '{output_type}'\n") from ae
    
    try:
        assert os.path.isfile( path_to_downsampled_file )
    except AssertionError as ae:
        raise FileNotFoundError( f"\npath_to_downsampled_file ({path_to_downsampled_file}) wasn't found\n" ) from ae

    # loading data, matrix creation
    downsampled_df = pds.read_csv( path_to_downsampled_file, **read_csv_params )
    downsampled_df[0] /= resolution 
    downsampled_df[1] /= resolution 
    downsampled_df = downsampled_df.astype( { 0: int, 1:int } )
    
    matrix_dims = 1 + max( downsampled_df[0].max(), downsampled_df[1].max() ) # the + 1 is to account for the 0-<res> bin
    hic_mat = np.zeros( ( matrix_dims, matrix_dims ) )
    print( hic_mat.shape )
    # populating matrix (in a triangular way)
    for row_tup in downsampled_df.itertuples():
        # row_tup: tuple of ( index, col1 value, col2 value, col3 value, col4 value, ... )
        
        if row_tup[1] <= row_tup[2]:
            hic_mat[ row_tup[1], row_tup[2] ] += row_tup[3]
        else:
            hic_mat[ row_tup[2], row_tup[1] ] += row_tup[3]
    
    if enforce_symm:
        upper_triangular_sum = np.sum( np.triu( hic_mat, 1 ) )
        lower_triangular_sum = np.sum( np.tril( hic_mat, -1 ) )
        # ensure the previous populating for-loop built a triangular matrix
        assert ( lower_triangular_sum == 0.0 ) or ( upper_triangular_sum == 0.0 )
        if upper_triangular_sum != lower_triangular_sum:
            hic_mat = triangularize_matrix( hic_mat )

    # saving output
    if output_type == "pandas":
        output = pds.DataFrame( hic_mat ) # fix this
        if as_pickle:
            with open( output_filename, 'wb' ) as pickle_handle:
                pickle.dump( output, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )
        else:
            # quick ref for float_format: https://docs.python.org/3/library/string.html#format-specification-mini-language
            output.to_csv( output_filename, sep='\t', float_format='%.3f' ) # header=output.columns, index=output.index, float_format='%.3f' ) 

    elif output_type == "sparse":
        output = csr_matrix( hic_mat )
        with open( output_filename, 'wb' ) as pickle_handle:
            pickle.dump( output, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )

    else: # output_type == "numpy"
        if as_pickle:
            with open( output_filename, 'wb' ) as pickle_handle:
                pickle.dump( hic_mat, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )
        else:
            np.savetxt( output_filename, hic_mat, delimiter='\t', fmt='%1.3f' ) # fix fmt argument


if __name__ == '__main__':
    create_matrix( 
        "test_file_for_matrix_construction.tsv",
        {
            'sep':'\t',
            'header':None,
            'index_col':None
        },
        "sparse",
        "test_output.tsv",
        enforce_symm=True
    )