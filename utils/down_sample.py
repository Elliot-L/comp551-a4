import pandas as pd
import random, os
from tqdm import tqdm 

def downsample_matrix(csv_path, read_csv_params, output_file_path, ratio=16, seed=7, verbose=False):
    """
    :param csv_path: path to the input file
    :param read_csv_params: arguments for the pd read_csv function
    :param ratio: downsampling ratio
    :param output_file_path: name of the output file, will have the see appended for reproducibility
    :param seed: random seed for random choice
    :return: No return, saves tsv file with downsampled matrix
    """
    df = pd.read_csv(csv_path, **read_csv_params)
    counts = {(row[1], row[2]): row[3] for row in df.itertuples()}  # row[0] is the index
    if verbose:
        print( "done reading file")
    pop = list()
    for key, val in tqdm( counts.items() ):
        pop.extend([key] * int(val))
    random.seed(seed)
    # there's probably a more efficient way to do this
    downsampled_pop = random.sample(pop, int(len(pop) / ratio))
    downsampled_counts = dict()
    for tup in tqdm( downsampled_pop ):
        if tup not in downsampled_counts.keys():
            downsampled_counts[tup] = 1
        else:
            downsampled_counts[tup] += 1
    if verbose:
        print( "done downsampling" )
    # making a pandas dataframe
    result_df = pd.DataFrame( [ [ key[0], key[1], val ] for key, val in downsampled_counts.items() ] )
    if verbose:
        print( "made downsampled dataframe" )
    name = output_file_path + '_' + str(seed) + '.gz'
    result_df.to_csv(name, sep='\t', header=False, index=False, compression='gzip')

    print('done')

def downsample_all_matrices( dirpath, num_downsamples=1 ):
    """
    Wrapper around downsample_matrix to iterate over all the raw files in a folder.

    Arguments:

        dirpath: path to the directory containing the raw .gz files (and no other files).

        num_downsamples: int indicating how many _separate_ downsample runs to make.

    Returns:

        Nothing, but saves each chromosome's downsampled .gz files in a folder called "downsampled_with_seed_#"

    """

    # argument validation
    assert os.path.isdir( dirpath )
    all_files = [ os.path.join( dirpath, f ) for f in os.listdir( dirpath ) ]
    
    for rand_seed in range( 1, 1+num_downsamples ):
        downsample_dir_name = os.path.join( dirpath, f"downsampled_with_seed_{rand_seed}" )
        os.makedirs( downsample_dir_name )
        for raw_file in all_files:
            print( os.path.basename( raw_file ) )
            downsample_matrix(
                raw_file,
                {
                    'sep':'\t',
                    'header':None,
                    'index_col':None,
                    'compression':'gzip'
                },
                os.path.join(
                    downsample_dir_name,
                    os.path.basename( raw_file )
                ),
                seed=rand_seed
            )

if __name__ == '__main__':
    '''
    downsample_matrix(
        "test_file_for_matrix_construction.tsv",
        {
            'sep':'\t',
            'header':None,
            'index_col':None
        },
        'downsample'
    )
    '''
    downsample_all_matrices(
        r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\raw",
        num_downsamples=2
    )
    
    
