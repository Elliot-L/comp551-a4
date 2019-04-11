import pickle
def unpickle_data_pickle( path_to_pickled_file ):

    with open( path_to_pickled_file, 'rb' ) as pickle_handle:
        list_of_arr, list_of_targets, list_of_chroms = zip( *pickle.load( pickle_handle ) )
    
    return list_of_arr, list_of_targets, list_of_chroms

