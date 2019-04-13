import os, argparse, pickle
import numpy as np
import pandas as pds

from scipy.ndimage import convolve

from stats.score import score_correlation_or_mse
from stats.avg_with_convolution import make_window_avgd_matrix
from stats.gaussian_blur_main import gaussian_blur, gkern, my_make_gaussian_kernel

from utils.iterate_over_diagonal import kth_diag_indices
from utils.sparse_matrix_construction import create_matrix

def assess_baselines( raw_data_dir, downsampled_data_dir, chromosome_range=list( range( 1,23 ) ), diag_offset_range=list( range( 0, 100 ) ), verbose=True, bandwidth_or_upper_triangular_correlation_output_file=None, diagonal_correlation_output_file=None, entire_triangular_comparison=False, metric='cor', rescale_factor=1.0, chrom_wise_pickle=True ):
    """
    Documentation TBD, use the off-tabbed # comments to collapse sections.
    """ 
# argument validation
    metric = metric.lower()
    try: 
        assert metric in [ "mse", "cor" ]
    except AssertionError as ae:
        raise AssertionError( f"\nthe 'metric' parameter can be one of 'mse' or 'cor'; you passed {metric}\n" )

    downsample_random_seed = downsampled_data_dir.split('_')[-1]
    
# saving metadata in output files (1/2)
    if bandwidth_or_upper_triangular_correlation_output_file is not None:
        with open( bandwidth_or_upper_triangular_correlation_output_file, 'w' ) as output_file_handle:
            output_file_handle.write( f"#raw_data_dir: {raw_data_dir.__repr__()}\n#downsampled_data_dir: {downsampled_data_dir.__repr__()}\n#downsample_random_seed: {downsample_random_seed}\n#chromosome_range: [{min(chromosome_range)},{max(chromosome_range)}]\n#diag_offset_range: [{min(diag_offset_range)},{max(diag_offset_range)}]\n#entire_triangular_comparison: {entire_triangular_comparison}\n#rescale_factor: {rescale_factor}\n" )
    
# saving metadata in output files (2/2)
    if diagonal_correlation_output_file is not None:
        with open( diagonal_correlation_output_file, 'w' ) as output_file_handle:
            output_file_handle.write( f"#raw_data_dir: {raw_data_dir.__repr__()}\n#downsampled_data_dir: {downsampled_data_dir.__repr__()}\n#downsample_random_seed: {downsample_random_seed}\n#chromosome_range: [{min(chromosome_range)},{max(chromosome_range)}]\n#diag_offset_range: [{min(diag_offset_range)},{max(diag_offset_range)}]\n#entire_triangular_comparison: {entire_triangular_comparison}\n#rescale_factor: {rescale_factor}\n" )

# computational block
    for chrom_number in chromosome_range:
        
        if verbose: print( f"\nstarting with chromosome {chrom_number}\n" )
    
    # update output files
        if bandwidth_or_upper_triangular_correlation_output_file is not None:
            with open( bandwidth_or_upper_triangular_correlation_output_file, 'a' ) as output_file_handle:
                output_file_handle.write( f"\n == chromosome {chrom_number} == \n" )
        
        if diagonal_correlation_output_file is not None:
            with open( diagonal_correlation_output_file, 'a' ) as output_file_handle:
                output_file_handle.write( f"\n == chromosome {chrom_number} == \n" )

    # loading matrix file contents into memory
        raw_chr_file = os.path.join( raw_data_dir, f"chr{chrom_number}_10kb.RAWobserved.gz" )
        downsampled_chr_file = os.path.join( downsampled_data_dir, f"chr{chrom_number}_10kb.RAWobserved.gz_{downsample_random_seed}.tsv.gz" )

        if verbose: print( f"loading matrices..." )
    # create upper triangular matrices
        raw_mat = create_matrix( 
            raw_chr_file, 
            read_csv_params={
                'sep':'\t',
                'header':None,
                'index_col':None,
                'compression':'gzip'
            },
            output_type="numpy",
            output_file_path=None
        )

        down_samp_mat = create_matrix( 
            downsampled_chr_file,
            read_csv_params={
                'sep':'\t',
                'header':None,
                'index_col':None,
                'compression':'gzip'
            },
            output_type="numpy",
            output_file_path=None
        )

        if rescale_factor != 1.0:
            sum_before_rescale = np.sum( down_samp_mat )
            down_samp_mat *= rescale_factor
            sum_after_rescale = np.sum( down_samp_mat )

            if bandwidth_or_upper_triangular_correlation_output_file is not None:
                with open( bandwidth_or_upper_triangular_correlation_output_file, 'a' ) as output_file_handle:
                    output_file_handle.write( f"\n == chromosome {chrom_number}'s sum before rescaling by {rescale_factor}: {sum_before_rescale}; after rescaling: {sum_after_rescale} == \n" )
        
            if diagonal_correlation_output_file is not None:
                with open( diagonal_correlation_output_file, 'a' ) as output_file_handle:
                    output_file_handle.write( f"\n == chromosome {chrom_number}'s sum before rescaling by {rescale_factor}: {sum_before_rescale}; after rescaling: {sum_after_rescale} == \n" )

        if verbose: print( f"loaded matrices.\nsanity-checking dimensions..." )

    # ensuring the matrices are of the same dimensions
        if raw_mat.shape != down_samp_mat.shape:
            largest_dim = max( raw_mat.shape[0], down_samp_mat.shape[0] )
            if down_samp_mat.shape[0] != largest_dim:
                down_samp_mat = np.pad( down_samp_mat, ( ( 0, largest_dim-down_samp_mat.shape[0] ), ( 0, largest_dim-down_samp_mat.shape[0] ) ), 'constant', constant_values=0 )
            else: # down_samp_mat is larger than raw_mat (unlikely)
                raw_mat = np.pad( raw_mat, ( ( 0, largest_dim-raw_mat.shape[0] ), ( 0, largest_dim-raw_mat.shape[0] ) ), 'constant', constant_values=0 )
        assert raw_mat.shape == down_samp_mat.shape 
        
        if verbose: print( f"sanity-checked dimensions.\nfetching bandwidth indices..." )
        
        diagonal_wise_correlations = pds.DataFrame(
            np.zeros( ( len( diag_offset_range ), 10 ), dtype=float ),
            index=[ f"diag #{i}" for i in diag_offset_range ],
            columns=[ "downsampled P", "downsampled S", "window-averaged P", "window-averaged S", "scipy-gaussian-blurred P", "scipy-gaussian-blurred S", "their gaussian-like-blurred P", "their gaussian-like-blurred S", "our gaussian-kernel P", "our gaussian-kernel S" ]
        )

        if metric == "mse":
            diagonal_wise_correlations.columns = [ "downsampled mse", "downsampled rmse", "window-averaged mse", "window-averaged rmse", "scipy-gaussian-blurred mse", "scipy-gaussian-blurred rmse", "their gaussian-like-blurred mse", "their gaussian-like-blurred rmse", "our gaussian-kernel mse", "our gaussian-kernel rmse" ]

    # finding the row and column indices of the bandwidth region
        bandwidth_row_inds, bandwidth_col_inds = [],[]
        diagonal_wise_indices = {}

        for diag_offset in diag_offset_range: 

            diag_row_inds, diag_col_inds = kth_diag_indices( down_samp_mat, diag_offset )
            bandwidth_row_inds.extend( diag_row_inds )
            bandwidth_col_inds.extend( diag_col_inds )
            diagonal_wise_indices[ diag_offset ] = diag_row_inds, diag_col_inds

        # recall that the matrices returned by create_matrix are either upper triangular matrices or symmetric matrices
        if verbose: print( f"fetched bandwidth indices.\ncalculating no-transform correlations..." )
        
    # downsample vs non-downsampled
        skip_upper_triangular_correlations = False 
        no_transform_matrix_correlations = ( None, None )
        if entire_triangular_comparison:
            try:
                if metric == 'cor':
                    no_transform_matrix_correlations = ( 
                        score_correlation_or_mse( raw_mat, down_samp_mat, triangular="upper", metric="pearson" )[0], 
                        score_correlation_or_mse( raw_mat, down_samp_mat, triangular="upper", metric="spearman" )[0] 
                    )
                else: # metric == 'mse'
                    no_transform_matrix_correlations = score_correlation_or_mse( raw_mat, down_samp_mat, triangular="upper", metric="mse" )

            except MemoryError:
                print( f"skipping correlation scores for chromosome {chrom_number}'s upper matrix due to memory error." )
                skip_upper_triangular_correlations = True 

        no_transform_bandwidth_correlations = None
        if metric == 'cor':
            no_transform_bandwidth_correlations = (  
                score_correlation_or_mse( raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="pearson" )[0], 
                score_correlation_or_mse( raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="spearman" )[0] 
            )
        else: # metric == 'mse'
            no_transform_bandwidth_correlations = score_correlation_or_mse( raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="mse" )

        for diag_offset in diag_offset_range:
            if metric == 'cor':
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "downsampled P" ] = score_correlation_or_mse( raw_mat[ diagonal_wise_indices[ diag_offset ] ], down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="pearson" )[0]
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "downsampled S" ] = score_correlation_or_mse( raw_mat[ diagonal_wise_indices[ diag_offset ] ], down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="spearman" )[0]
            else: # metric == 'mse'
                mse, rmse = score_correlation_or_mse( raw_mat[ diagonal_wise_indices[ diag_offset ] ], down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="mse" )
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "downsampled mse" ] = mse 
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "downsampled rmse" ] = rmse 

        if verbose: print( f"calculated no-transform correlations.\ncalculating window-averaged correlations..." )
        
    # window-averaged downsample vs non-downsampled
        wavg_raw_mat = make_window_avgd_matrix( raw_mat, 3, ret_type=raw_mat.dtype )
        wavg_down_samp_mat = make_window_avgd_matrix( down_samp_mat, 3, ret_type=down_samp_mat.dtype )
        wavg_matrix_correlations = ( None, None )
        if entire_triangular_comparison and ( not skip_upper_triangular_correlations ) :
            if metric == 'cor':
                wavg_matrix_correlations = ( 
                    score_correlation_or_mse( wavg_raw_mat, wavg_down_samp_mat, triangular="upper", metric="pearson" )[0], 
                    score_correlation_or_mse( wavg_raw_mat, wavg_down_samp_mat, triangular="upper", metric="spearman" )[0] 
                )
            else: # metric == 'mse'
                wavg_matrix_correlations = score_correlation_or_mse( wavg_raw_mat, wavg_down_samp_mat, triangular="upper", metric="mse" )

        wavg_bandwidth_correlations = None 
        if metric == 'cor':
            wavg_bandwidth_correlations = ( 
                score_correlation_or_mse( wavg_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], wavg_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="pearson" )[0], 
                score_correlation_or_mse( wavg_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], wavg_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="spearman" )[0] 
            )
        else: # metric == 'mse'
            wavg_bandwidth_correlations = score_correlation_or_mse( wavg_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], wavg_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="mse" )

        for diag_offset in diag_offset_range:
            if metric == 'cor':
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "window-averaged P" ] = score_correlation_or_mse( wavg_raw_mat[ diagonal_wise_indices[ diag_offset ] ], wavg_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="pearson" )[0]
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "window-averaged S" ] = score_correlation_or_mse( wavg_raw_mat[ diagonal_wise_indices[ diag_offset ] ], wavg_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="spearman" )[0]
            else: # metric == 'mse'
                mse, rmse = score_correlation_or_mse( wavg_raw_mat[ diagonal_wise_indices[ diag_offset ] ], wavg_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="mse" )
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "window-averaged mse" ] = mse
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "window-averaged rmse" ] = rmse

        del wavg_raw_mat, wavg_down_samp_mat
        
        if verbose: print( f"calculated window-averaged correlations.\ncalculating scipy-gb correlations..." )

    # scipy-gaussian-blurred-downsample vs non-downsampled
        scipy_gb_raw_mat = gaussian_blur( raw_mat, "scipy", gaussian_blur_parameters={'sigma':4,'mode':'nearest'} )
        scipy_gb_down_samp_mat = gaussian_blur( down_samp_mat, "scipy", gaussian_blur_parameters={'sigma':4,'mode':'nearest'} )
        
        scipy_gb_matrix_correlations = ( None, None )
        if entire_triangular_comparison and (not skip_upper_triangular_correlations ):
            if metric == 'cor':
                scipy_gb_matrix_correlations = ( 
                    score_correlation_or_mse( scipy_gb_raw_mat, scipy_gb_down_samp_mat, triangular="upper", metric="pearson" )[0], 
                    score_correlation_or_mse( scipy_gb_raw_mat, scipy_gb_down_samp_mat, triangular="upper", metric="spearman" )[0] 
                )
            else: # metric == 'mse'
                scipy_gb_matrix_correlations = score_correlation_or_mse( scipy_gb_raw_mat, scipy_gb_down_samp_mat, triangular="upper", metric="mse" )

        scipy_gb_bandwidth_correlations = None
        if metric == 'cor':
            scipy_gb_bandwidth_correlations = ( 
                score_correlation_or_mse( scipy_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], scipy_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="pearson" )[0], 
                score_correlation_or_mse( scipy_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], scipy_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="spearman" )[0]
            )
        else: # metric == 'mse'
            scipy_gb_bandwidth_correlations = score_correlation_or_mse( scipy_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], scipy_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="mse" )

        for diag_offset in diag_offset_range:
            if metric == 'cor':
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "scipy-gaussian-blurred P" ] = score_correlation_or_mse( scipy_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], scipy_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="pearson" )[0]
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "scipy-gaussian-blurred S" ] = score_correlation_or_mse( scipy_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], scipy_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="spearman" )[0]
            else: # metric == 'mse'
                mse, rmse = score_correlation_or_mse( scipy_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], scipy_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="mse" )
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "scipy-gaussian-blurred mse" ] = mse 
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "scipy-gaussian-blurred rmse" ] = rmse

        del scipy_gb_raw_mat, scipy_gb_down_samp_mat
        if verbose: print( f"calculated scipy-gb correlations.\ncalculating incorrect-gb correlations..." )

    # their gaussian-like-blurred-downsample vs non-downsample
        incorrect_gauss_kern = gkern( 13, 4 )
        incorrect_gb_raw_mat = convolve( raw_mat, incorrect_gauss_kern, mode='nearest' )
        incorrect_gb_down_samp_mat = convolve( down_samp_mat, incorrect_gauss_kern, mode='nearest' )
        incorrect_gb_matrix_correlations = ( None, None )
        if entire_triangular_comparison and ( not skip_upper_triangular_correlations ):
            if metric == 'cor':
                incorrect_gb_matrix_correlations = ( 
                    score_correlation_or_mse( incorrect_gb_raw_mat, incorrect_gb_down_samp_mat, triangular="upper", metric="pearson" )[0], 
                    score_correlation_or_mse( incorrect_gb_raw_mat, incorrect_gb_down_samp_mat, triangular="upper", metric="spearman" )[0] 
                )
            else: # metric == 'mse'
                incorrect_gb_matrix_correlations = score_correlation_or_mse( incorrect_gb_raw_mat, incorrect_gb_down_samp_mat, triangular="upper", metric="mse" )
        
        incorrect_gb_bandwidth_correlations = None
        if metric == 'cor':
            incorrect_gb_bandwidth_correlations = ( 
                score_correlation_or_mse( incorrect_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], incorrect_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="pearson" )[0], 
                score_correlation_or_mse( incorrect_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], incorrect_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="spearman" )[0] 
            )
        else: # metric == 'mse'
            incorrect_gb_bandwidth_correlations = score_correlation_or_mse( incorrect_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], incorrect_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="mse" )

        for diag_offset in diag_offset_range:
            if metric == 'cor':
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "their gaussian-like-blurred P" ] = score_correlation_or_mse( incorrect_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], incorrect_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="pearson" )[0]
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "their gaussian-like-blurred S" ] = score_correlation_or_mse( incorrect_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], incorrect_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="spearman" )[0]
            else: # metric == 'mse'
                mse, rmse = score_correlation_or_mse( incorrect_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], incorrect_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="mse" )
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "their gaussian-like-blurred mse" ] = mse 
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "their gaussian-like-blurred rmse" ] = rmse

        del incorrect_gb_raw_mat, incorrect_gb_down_samp_mat
        
        if verbose: print( f"calculated incorrect-gb correlations.\ncalculating homebrewed gbk correlations..." )
        
    # our gaussian-kernel-convolution-blurred-downsample vs non-downsampled
        better_gauss_kern = my_make_gaussian_kernel( 13, [ 0.0, 0.0 ], np.eye(2)*4.0, normalize=True )
        better_gb_raw_mat = convolve( raw_mat, better_gauss_kern, mode='nearest' )
        better_gb_down_samp_mat = convolve( down_samp_mat, better_gauss_kern, mode='nearest' )
        better_gb_matrix_correlations = ( None, None )
        if entire_triangular_comparison and ( not skip_upper_triangular_correlations ):
            if metric == 'cor':
                better_gb_matrix_correlations = ( 
                    score_correlation_or_mse( better_gb_raw_mat, better_gb_down_samp_mat, triangular="upper", metric="pearson" )[0], 
                    score_correlation_or_mse( better_gb_raw_mat, better_gb_down_samp_mat, triangular="upper", metric="spearman" )[0]
                )
            else: # metric == 'mse'
                better_gb_matrix_correlations = score_correlation_or_mse( better_gb_raw_mat, better_gb_down_samp_mat, triangular="upper", metric="mse" )
        
        better_gb_bandwidth_correlations = None
        if metric == 'cor':
            better_gb_bandwidth_correlations = (  
                score_correlation_or_mse( better_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], better_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="pearson" )[0], 
                score_correlation_or_mse( better_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], better_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="spearman" )[0] 
            )
        else: # metric == 'mse'
            better_gb_bandwidth_correlations = score_correlation_or_mse( better_gb_raw_mat[ bandwidth_row_inds, bandwidth_col_inds ], better_gb_down_samp_mat[ bandwidth_row_inds, bandwidth_col_inds ], triangular="already flattened", metric="mse" )

        for diag_offset in diag_offset_range:
            if metric == 'cor':
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "our gaussian-like-blurred P" ] = score_correlation_or_mse( better_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], better_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="pearson" )[0]
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "our gaussian-like-blurred S" ] = score_correlation_or_mse( better_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], better_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="spearman" )[0]
            else: # metric == mse
                mse, rmse = score_correlation_or_mse( better_gb_raw_mat[ diagonal_wise_indices[ diag_offset ] ], better_gb_down_samp_mat[ diagonal_wise_indices[ diag_offset ] ], triangular="already flattened", metric="mse" )
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "our gaussian-like-blurred mse" ] = mse
                diagonal_wise_correlations.loc[ f"diag #{diag_offset}", "our gaussian-like-blurred rmse" ] = rmse

        del better_gb_raw_mat, better_gb_down_samp_mat

    # appending this chromosome's results to output files
        correlations_dataframe = pds.DataFrame(
            np.array(
                [
                    # first row
                    [ 
                        no_transform_matrix_correlations[0], no_transform_matrix_correlations[1], wavg_matrix_correlations[0], wavg_matrix_correlations[1], scipy_gb_matrix_correlations[0], scipy_gb_matrix_correlations[1], incorrect_gb_matrix_correlations[0], incorrect_gb_matrix_correlations[1], better_gb_matrix_correlations[0], better_gb_matrix_correlations[1]
                    ],
                    # second row
                    [ 
                        no_transform_bandwidth_correlations[0], no_transform_bandwidth_correlations[1], wavg_bandwidth_correlations[0], wavg_bandwidth_correlations[1], scipy_gb_bandwidth_correlations[0], scipy_gb_bandwidth_correlations[1], incorrect_gb_bandwidth_correlations[0], incorrect_gb_bandwidth_correlations[1], better_gb_bandwidth_correlations[0], better_gb_bandwidth_correlations[1]
                    ]
                ]
            ),
            columns=["down-sample P", "down-sample S", "window-averaged down-sample P", "window-averaged down-sample S", "scipy GB down-sample P", "scipy GB down-sample S", "their GBK down-sample P", "their GBK down-sample S", "our GBK down-sample P", "our GBK down-sample S" ],
            index=["on entire triangular", "on bandwidth region"]
        )

        if metric == 'mse':
            correlations_dataframe.columns = ["down-sample mse", "down-sample rmse", "window-averaged down-sample mse", "window-averaged down-sample rmse", "scipy GB down-sample mse", "scipy GB down-sample rmse", "their GBK down-sample mse", "their GBK down-sample rmse", "our GBK down-sample mse", "our GBK down-sample rmse" ]

        if verbose: print( correlations_dataframe )

        if bandwidth_or_upper_triangular_correlation_output_file is not None:
            correlations_dataframe.to_csv( bandwidth_or_upper_triangular_correlation_output_file, mode='a', sep='\t', float_format='%1.3f' )
            print( f"saved triangular/bandwidth correlations for chromosome {chrom_number} under:\n{bandwidth_or_upper_triangular_correlation_output_file}\n")
            if chrom_wise_pickle:
                with open( f"{bandwidth_or_upper_triangular_correlation_output_file.split('.')[0]}_chrom_{chrom_number}_data.pickle", "wb" ) as pickle_handle:
                    pickle.dump( correlations_dataframe, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )

        if diagonal_correlation_output_file is not None:
            diagonal_wise_correlations.to_csv( diagonal_correlation_output_file, mode='a', sep='\t', float_format='%1.3f' )
            print( f"saved diagonal correlations for chromosome {chrom_number} under:\n{diagonal_correlation_output_file}\n")
            if chrom_wise_pickle:
                with open( f"{diagonal_correlation_output_file.split('.')[0]}_chrom_{chrom_number}_data.pickle", "wb" ) as pickle_handle:
                    pickle.dump( diagonal_wise_correlations, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL )
                    
        

        # prior kernel ?
    print( "\nfinished." )

if __name__ == '__main__':
#for debugging/testing functions
    #assess_baselines( 
        # arguments used for debugging
        #r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\test", 
        #r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\test", 
        # actual paths
    #    r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\raw", 
    #    r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\comp551-a4\data\raw\downsampled_with_seed_1",
    #    chromosome_range=list( reversed( range( 3,23 ) ) ),
    #    bandwidth_or_upper_triangular_correlation_output_file=os.path.join( os.getcwd(), "bandwidth_test_for_mse_rmse_output.tsv" ),
    #    diagonal_correlation_output_file=os.path.join( os.getcwd(), "diagonal-wise_test_for_mse_rmse_output.tsv" ),
    #    metric="mse"
    #)

    parser = argparse.ArgumentParser( description='Evaluating baselines' )
    parser.add_argument( '--raw-data-dir', type=str, metavar='R', required=True,
                        help='path to the directory containing the raw data files.' )
    parser.add_argument( '--downsampled-data-dir', type=str, metavar='D', required=True,
                        help='path to the directory containing the downsampled data files.' )
    parser.add_argument( '--skip-chrom', type=int, nargs='+', metavar='C', default=None,
                        help='space-delimited list of chromosomes to skip.' )
    parser.add_argument( '--metric', type=str, metavar='M', default="cor", required=True,
                        help='metric type to use ("mse" or "cor") (default="cor").' )
    parser.add_argument( '--bandwidth-output-file-path', type=str, metavar='B', required=True,
                        help="path for the output file containing bandwidth-wise scoring." )
    parser.add_argument( '--diagonal-output-file-path', type=str, metavar='D', required=True,
                        help="path for the output file containing diagonal-wise scoring." )
    parser.add_argument( '--rescale-factor-for-mse', type=float, metavar='F', default=1.0,
                        help="factor (float) used to rescale the entries in the downsampled matrix before doing mse & rmse comparisons." )

    args = parser.parse_args()

    chrom_range = list( range( 1,23 ) )[::-1] 
    if args.skip_chrom is not None:
        chrom_range = list( [ chrom_num for chrom_num in range( 1, 23 ) if chrom_num not in args.skip_chrom ] )[::-1]

    assess_baselines(
        args.raw_data_dir,
        args.downsampled_data_dir,
        chromosome_range=chrom_range,
        bandwidth_or_upper_triangular_correlation_output_file=args.bandwidth_output_file_path,
        diagonal_correlation_output_file=args.diagonal_output_file_path,
        metric=args.metric,
        rescale_factor=args.rescale_factor_for_mse
    )

    # example: if I wanted to go over all chromosomes except 1,2,3, I would run
    # python score_baselines.py --raw-data-dir path/to/raw/data/dir --downsampled-data-dir path/to/downsampled/data/dir --skip-chrom 1 2 3 --metric mse --bandwidth-output-file-path path/to/output/file1.tsv --diagonal-output-file-path path/to/output/file2.tsv

