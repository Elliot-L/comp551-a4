# comp551-a4
final project for comp 551

knn_regressor.py
functions: train_and_test
description: trains and evaluates a KNN model on specified datasets

logger.py
classes: Logger
description: used for recording data while training pytorch models

make_training_data.py
functions: make_training_data_list, make_training_data_for_downsampled_dir
description: constructs a dataset given input downsampled matrices and target full resolution matrices

pull_from_modd_run_barebones.py
functions: train, validate
description: performs training and validation of a specified pytorch model architecture


random_forest_regressor.py
functions: train_and_test
description: trains and evaluates a RF model on specified datasets


run_on_test_set.py
functions: run_on_test
description: loads a saved pytorch model and evaluates its performance on a specified dataset without training


score_baselines.py
functions: assess_baselines
description: perferms the full evaluation of image analysis baseline methods


utils/down_sample.py
functions: downsample_matrix, downsample_all_matrices
description: for doing the random downsampling of the full resolution target matrices


utils/unpickle.py
functions: unpickle_data_pickle, gather_chromosome_data
description: for assembling pickle files of data


utils/iterate_over_diagonal.py
functions: kth_diag_indices, iterate_over_diagonals, get_diagonal_wise_windows, make_mat_with_only_k_diags
description: For repeatedly running over diagonals across a large matrix (in order to extract submatrices)

utils/sparse_matrix_reconstruction.py
functions: negate_triangular, make_matrix_symmetric, create_matrix
description: utilities for assembling a sparse matrix


stats/avg_with_convolutions.py
functions: make_window_avgd_matrix
description: transforms an input matrix by averaging it using a specified window

stats/guassian_blue_main.py
functions: my_make_gaussian_kernel, gaussian_blur, gkern, Gaussian_filter, compare_gaussian_kernels
description: stats for performing the guassian blurring over HiC matrices

stats/score.py
functions: score_correlation_or_mse
description: used to evaluate the baseline image analysis outputs

models/cnn_models.py
classes: RNN13, BaseNet, BaseNetPlus, ThreeLayerModel13, ThreeLayerModel40
description: classes defining various neural network architectures

models/simple_test.py
functions: main
description: Used to load dummy data into a neural network for debugging