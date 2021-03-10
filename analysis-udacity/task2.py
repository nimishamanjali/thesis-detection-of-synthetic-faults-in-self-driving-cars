import os
import sys
from functools import reduce

import numpy as np
import pandas as pd

from analyse_crashes_obes import extract_crashes_obes_data, build_mutant_list_not_having_crashes_obes, \
    build_mutant_list_having_crashes_obes_on_some_models
from stats import is_diff_sts

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

STD_SPEED = 'Std(Speed)'
MEAN_LP = 'Mean(LP)'
STD_SA = 'Std(SA)'
MAX_LP = 'Max(LP)'
# STD_BRAKE = 'Std(Brake)'
MAX_ACC = 'Max(Acc)'
metrics = [MEAN_LP, STD_SPEED, STD_SA, MAX_LP, MAX_ACC]


# given list of mutants with no crashes or obes ,
#  it extracts the metrics mean_LP,std_sa,std_speed of those mutant models to a dataframe
def extract_data_based_given_mutant_list(path, no_crashes_obes_list):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith(tuple(no_crashes_obes_list))]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    mean_lp = []
    std_sa = []
    std_speed = []
    max_lp = []
    max_acc = []
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=metrics)
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_lp.append((filtered_csv[MEAN_LP].tolist()))
        std_sa.append((filtered_csv[STD_SA].tolist()))
        std_speed.append((filtered_csv[STD_SPEED].tolist()))
        max_lp.append(filtered_csv[MAX_LP].tolist())
        max_acc.append(filtered_csv[MAX_ACC].tolist())

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed,
         MAX_LP: max_lp,
         MAX_ACC: max_acc
         })


# extract and save the original model data to a dataframe
def extract_original_model_data(path):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    mean_lp = []
    std_sa = []
    std_speed = []
    max_lp = []
    max_acc = []

    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=metrics)
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_lp.append(filtered_csv[MEAN_LP].tolist())
        std_sa.append(filtered_csv[STD_SA].tolist())
        std_speed.append(filtered_csv[STD_SPEED].tolist())
        max_lp.append(filtered_csv[MAX_LP].tolist())

        max_acc.append(filtered_csv[MAX_ACC].tolist())

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed,
         MAX_LP: max_lp,
         MAX_ACC: max_acc
         })


def remove_this_func_later(a, i):
    try:
        [item[i] for item in a]
    except IndexError:
        return False
    return True


# perform statistical def of killing comparing each 20 mutant version to each 20 original model
def compute_table_for_metric(org, x, metric):
    killed = None
    if len(x[metric]) == 20:
        mutant_list = x[metric].tolist()
        org_list = org[metric].tolist()
        for i in range(0, 27):
            if remove_this_func_later(mutant_list, i):

                data = is_diff_sts([item[i] for item in org_list], [item[i] for item in mutant_list])

                if data[0]:
                    killed = 'killed: ' + str(data[0]) + ',' + 'Sector number: ' + str(i)
                    break

    return tuple(killed.split(",")) if killed is not None else 'killed: false'


def print_killed_mutants(df):
    merged = []
    for i in metrics:
        killed_mutants = filter_killed_mutants(df, i)
        merged.append(killed_mutants)
    print(reduce(lambda df1, df2: pd.merge(df1, df2, on='mutation', how='outer'), merged).replace(np.nan,
                                                                                                  'Not killed by this metric',
                                                                                                  regex=True))


def filter_killed_mutants(df, metric):
    return df.loc[df[metric] != 'killed: false'][['mutation', metric]]


def compute_and_merge_metric_mutant_tables(df, org_model_data):
    mean_lp_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, MEAN_LP)).reset_index(
        name=MEAN_LP)

    std_sa_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, STD_SA)).reset_index(
        name=STD_SA)
    std_speed_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, STD_SPEED)).reset_index(
        name=STD_SPEED)
    max_lp_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, MAX_LP)).reset_index(
        name=MAX_LP)

    max_acc_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, MAX_ACC)).reset_index(
        name=MAX_ACC)
    return reduce(lambda left, right: pd.merge(left, right, on=['mutation']),
                  [mean_lp_stat_table, std_sa_stat_table, std_speed_stat_table, max_lp_stat_table, max_acc_stat_table])


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    df_crashes_obes = extract_crashes_obes_data(sys.argv[1])
    org_model_data = extract_original_model_data(sys.argv[1])

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df_crashes_obes)
    df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list)
    table_mutants_for_no_crashes_obes = compute_and_merge_metric_mutant_tables(df_no_crashes_obes,
                                                                               org_model_data)
    print(
        'mutants killed by metrics from mutant list not having crashes or obes_________')

    (print_killed_mutants(table_mutants_for_no_crashes_obes))

    some_crashes_obes_list = build_mutant_list_having_crashes_obes_on_some_models(df_crashes_obes)
    df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                some_crashes_obes_list['mutation'].tolist())
    table_mutants_for_some_crashes_obes = compute_and_merge_metric_mutant_tables(df_some_crashes_obes,
                                                                                 org_model_data)
    print(
        '\n mutants killed by metrics from mutant list having some crashes or obes_________')
    (print_killed_mutants(table_mutants_for_some_crashes_obes))
