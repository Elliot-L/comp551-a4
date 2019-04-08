import random, os, gzip, io, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm 

def downsample_matrix(csv_path, read_csv_params, output_file_path, ratio=16, seed=7, verbose=True):
    """
    :param csv_path: path to the input file
    :param read_csv_params: arguments for the pd read_csv function
    :param ratio: downsampling ratio
    :param output_file_path: name of the output file, will have the see appended for reproducibility
    :param seed: random seed for random choice
    :verbose: bool indicator of verbosity
    :return: No return, saves tsv file with downsampled matrix
    """
    df = pd.read_csv(csv_path, **read_csv_params)
    counts = {(row[1], row[2]): row[3] for row in df.itertuples()}  # row[0] is the index
    if verbose:
        print( "done reading file")
    pop = list()
    for key, val in tqdm( counts.items() ):
        pop.extend([key] * int(val))
    if verbose:
        print( "expanded contents has been made into a list, starting downsampling" )
    
    random.seed(seed)
    # there's probably a more efficient way to do this
    downsampled_pop = random.sample(pop, int(len(pop) / ratio))
    downsampled_counts = dict()
    for tup in tqdm( downsampled_pop ):
        try:
            downsampled_counts[tup] += 1
        except KeyError:
            downsampled_counts[tup] = 1
        '''
        the try/except block above is a faster way to do this / maybe, it fluctuates quite a bit on my laptop
        if tup not in downsampled_counts.keys():
            downsampled_counts[tup] = 1
        else:
            downsampled_counts[tup] += 1
        '''
    if verbose:
        print( f"done downsampling, dictionary has {len( downsampled_pop )} entries" )
    
    # if-block circumvents the slowness of pandas with large dataframes
    # chr1, chr2, and chr3 are the files that (empirically) gave pandas the hardest time
    if ( "chr1_" in output_file_path ) or ( "chr2_" in output_file_path ) or ( "chr3_" in output_file_path ):
        name = output_file_path + '_' + str(seed) + '.tsv.gz'
        # convert to \n-delimited stream 
        print( "making stream version of output data" )
        contents = '\n'.join( [ f"{key[0]}\t{key[1]}\t{val}" for key, val in downsampled_counts.items() ] )
        print( "saving stream to output file (may take ~5-7 minutes)" )
        with gzip.open( name, 'wb' ) as outfile:
            with io.TextIOWrapper( outfile, encoding='utf-8' ) as encoded:
                encoded.write( contents )
        del contents
    else:
        # making a pandas dataframe
        arr = np.array( [ [ key[0], key[1], val ] for key, val in downsampled_counts.items() ], dtype='uint32' )
        if verbose:
            print( "made downsampled array" )
        result_df = pd.DataFrame( arr )
        if verbose:
            print( "made downsampled dataframe" )
        name = output_file_path + '_' + str(seed) + 'tsv.gz'
        result_df.to_csv(name, sep='\t', header=False, index=False, compression='gzip')
        del result_df 
        del arr
    print('done')
    del pop 
    del downsampled_pop
    del df
    del counts

def downsample_all_matrices( dirpath, seeds_list=1 ):
    """
    Wrapper around downsample_matrix to iterate over all the raw files in a folder.

    Arguments:

        dirpath: path to the directory containing the raw .gz files (and no other files).

        seeds_list: list of ints representing random seeds.

    Returns:

        Nothing, but saves each chromosome's downsampled .gz files in a folder called "downsampled_with_seed_#"

    """

    # argument validation
    assert os.path.isdir( dirpath )
    all_files = list( reversed( [ os.path.join( dirpath, f ) for f in os.listdir( dirpath ) if os.path.isfile( os.path.join( dirpath, f ) ) ] ) )
    
    for rand_seed in seeds_list:
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
                    'compression':'gzip',
                    'dtype':'uint32'
                },
                os.path.join(
                    downsample_dir_name,
                    os.path.basename( raw_file )
                ),
                seed=rand_seed
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Downsampling Hi-C Dataset' )
    parser.add_argument('--raw-data-dir', type=str, metavar='P', required=True,
                        help='path to the directory containing the raw data files.')
    parser.add_argument('--random-seeds', type=int, nargs='+', metavar='S', required=True,
                        help='space-separated integers representing seeds for random downsampling')
    args = parser.parse_args()
    downsample_all_matrices(
        args.raw_data_dir,
        seeds_list=args.random_seeds
    )    

    # testing code for individual functions
    # to test downsample_matrix:
    #downsample_matrix(
    #    "test_file_for_matrix_construction.tsv",
    #    {
    #        'sep':'\t',
    #        'header':None,
    #        'index_col':None
    #    },
    #    'downsample'
    #)
    #################################
    # to test downsample_all_matrices:
    #downsample_all_matrices(
    #    r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\raw",
    #    seeds_list=[1234,99999]
    #)