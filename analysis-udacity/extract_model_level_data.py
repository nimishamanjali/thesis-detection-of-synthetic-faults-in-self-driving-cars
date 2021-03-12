import os
import sys
from pprint import pprint

import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)


def make_tuple_from_string(str):
    return tuple(str.split(' '))


def remove_if_nan(lst):
    return [i for i in lst if 'nan' not in i]


# 7.96
def extract_data_from_csv(path):
    binary_search_files = []
    exs_search_files = []
    no_search_files = []
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        for file in files:
            if file.endswith(".csv"):
                filenames.append(os.path.join(root, file))

    for i in filenames:
        model_name = '_'.join(
            (os.path.splitext(i.split('/')[-1])[0]).split('_')[:-1])
        if model_name == 'change_learning_rate':
            model_name = model_name + '_mutated0_MP_False_'
        elif model_name == 'change_epochs':
            model_name = model_name + '_mutated0_MP_50_'
        else:
            model_name = model_name + '_mutated0_MP_'

        if i.endswith('_binarysearch.csv'):

            tmp = pd.read_csv(i, header=None)
            if not tmp.empty:
                binary_search_files.append(model_name + str(
                    '%g' % (tmp.iloc[0, 1])) + ' ' + str((tmp.iloc[0, 5])))

                for index, row in tmp.iterrows():
                    if index != 0:
                        binary_search_files.append(model_name + str(
                            '%g' % (row[2])) + ' ' + str(row[5]))

        if i.endswith('_exssearch.csv'):
            tmp = pd.read_csv(i, header=None)
            if not tmp.empty:
                for index, row in tmp.iterrows():
                    exs_search_files.append(model_name + str(row[0]) + ' ' + str(row[3]))

        if i.endswith('_nosearch.csv'):
            tmp = pd.read_csv(i, header=None)
            no_search_files.append(model_name[:-1] + ' ' + str(tmp.iloc[0, -1]))

    final_list = binary_search_files + exs_search_files + no_search_files
    tuple_list = [make_tuple_from_string(element) for element in final_list]
    return remove_if_nan(tuple_list)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    pprint((extract_data_from_csv(sys.argv[1])))
