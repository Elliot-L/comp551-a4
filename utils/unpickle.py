import pickle
import os


def unpickle_data_pickle( path_to_pickled_file ):

    with open( path_to_pickled_file, 'rb' ) as pickle_handle:
        list_of_arr, list_of_targets, list_of_chroms = zip( *pickle.load( pickle_handle ) )
    
    return list_of_arr, list_of_targets, list_of_chroms


def gather_chromosome_data(data_dir):
    input_vector = list()
    output_vector = list()
    chroms_vector = list()

    for file in os.listdir(data_dir):
        if 'pickle' in file:
            inputs, outputs, chroms = unpickle_data_pickle(os.path.join(data_dir, file))
            input_vector.extend(inputs)
            output_vector.extend(outputs)
            chroms_vector.extend(chroms)

    return input_vector, output_vector, chroms_vector
