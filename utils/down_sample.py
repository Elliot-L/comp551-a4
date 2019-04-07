import pandas as pd
import random

def downsample_matrix(csv_path, read_csv_params, output_file_path, ratio=16, seed=7):
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

    pop = list()
    for key, val in counts.items():
        pop.extend([key] * int(val))
    random.seed(seed)
    # there's probably a more efficient way to do this
    downsampled_pop = random.sample(pop, int(len(pop) / ratio))
    downsampled_counts = dict()
    for tup in downsampled_pop:
        if tup not in downsampled_counts.keys():
            downsampled_counts[tup] = 1
        else:
            downsampled_counts[tup] += 1

    # making a pandas dataframe
    result_df = pd.DataFrame(columns=['tup0', 'tup1', 'count'])
    for tup, count in downsampled_counts.items():
        result_df = result_df.append({
            'tup0': tup[0],
            'tup1': tup[1],
            'count': count
            }, ignore_index=True)
    name = output_file_path + '_' + str(seed) + '.tsv'
    result_df.to_csv(name, sep='\t', header=False)

    print('done')

if __name__ == '__main__':
    downsample_matrix(
        "test_file_for_matrix_construction.tsv",
        {
            'sep':'\t',
            'header':None,
            'index_col':None
        },
        'downsample'
    )