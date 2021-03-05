import os
import sys
from pprint import pprint

import pandas as pd

from analyse_crashes_obes import extract_data, build_mutant_list_not_having_crashes_obes
from stats import is_diff_sts

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)


def get_data(path, l):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith(tuple(l))]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    mean_LP = []
    std_SA = []
    std_speed = []
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=['Mean(LP)', 'Std(SA)', 'Std(Speed)'])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_LP.append((filtered_csv.sum(axis=0)['Mean(LP)']))
        std_SA.append((filtered_csv.sum(axis=0)['Std(SA)']))
        std_speed.append((filtered_csv.sum(axis=0)['Std(Speed)']))
        # print(os.path.splitext(i.split('/')[-2])[0])

    return pd.DataFrame(
        {'mutation': filename,
         'Mean(LP)': mean_LP,
         'Std(SA)': std_SA,
         'Std(Speed)': std_speed
         })


def get_original(path):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    mean_LP = []
    std_SA = []
    std_speed = []
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=['Mean(LP)', 'Std(SA)', 'Std(Speed)'])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_LP.append((filtered_csv.sum(axis=0)['Mean(LP)']))
        std_SA.append((filtered_csv.sum(axis=0)['Std(SA)']))
        std_speed.append((filtered_csv.sum(axis=0)['Std(Speed)']))

    return pd.DataFrame(
        {'mutation': filename,
         'Mean(LP)': mean_LP,
         'Std(SA)': std_SA,
         'Std(Speed)': std_speed
         })


def compute_table_for_mean_lp(org, x):
    return is_diff_sts(x['Mean(LP)'].tolist(), org['Mean(LP)'].tolist())


def compute_table_for_std_sa(org, x):
    return is_diff_sts(x['Std(SA)'].tolist(), org['Mean(LP)'].tolist())


def compute_table_for_std_speed(org, x):
    return is_diff_sts(x['Std(Speed)'].tolist(), org['Std(Speed)'].tolist())


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    df = extract_data(sys.argv[1])
    l = build_mutant_list_not_having_crashes_obes(df)
    pprint(l)
    whole_df = (get_data(sys.argv[1], l))
    org = get_original(sys.argv[1])
    l1 = (whole_df.groupby(by=['mutation'])).apply(lambda x: compute_table_for_mean_lp(org, x)).reset_index(
        name='Mean(LP)')
    l2 = (whole_df.groupby(by=['mutation'])).apply(lambda x: compute_table_for_std_sa(org, x)).reset_index(
        name='Std(SA)')
    l3 = (whole_df.groupby(by=['mutation'])).apply(lambda x: compute_table_for_std_speed(org, x)).reset_index(
        name='Std(Speed)')
    print(l1.merge(l2, on='mutation').merge(l3, on='mutation'))
