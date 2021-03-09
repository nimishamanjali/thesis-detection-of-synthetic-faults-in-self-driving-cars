import os
import sys
from pprint import pprint

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
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=[MEAN_LP, STD_SA, STD_SPEED])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_lp.append((filtered_csv[MEAN_LP].tolist()))
        std_sa.append((filtered_csv[STD_SA].tolist()))
        std_speed.append((filtered_csv[STD_SPEED].tolist()))

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed
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
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=[MEAN_LP, STD_SA, STD_SPEED])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        mean_lp.append((filtered_csv[MEAN_LP].tolist()))
        std_sa.append((filtered_csv[STD_SA].tolist()))
        std_speed.append((filtered_csv[STD_SPEED].tolist()))

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed
         })


# perform statistical def of killing comparing each 20 mutant version to each 20 original model
def compute_table_for_metric(org, x, metric):
    killed = None
    if len(x[metric]) == 20:
        mutant_list = x[metric].tolist()
        org_list = org[metric].tolist()
        for i in range(0, 19):
            data = is_diff_sts(org_list[i], mutant_list[i])
            if data[0]:
                killed = 'killed: ' + str(data[0]) + '; p_value: ' + str(round(data[1], 3)) + '; effect_size: ' + str(
                    round(data[2], 3))
                break
    return str(killed) if killed is not None else 'killed: false'


def filter_killed_mutants(df):
    mutant_list = [(df.loc[df[MEAN_LP] != 'killed: false']['mutation'].tolist()),
                   (df.loc[df[STD_SA] != 'killed: false']['mutation'].tolist()),
                   (df.loc[df[STD_SPEED] != 'killed: false']['mutation'].tolist())]
    return list(set([item for sublist in mutant_list for item in sublist]))


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
    return mean_lp_stat_table.merge(std_sa_stat_table, on='mutation').merge(std_speed_stat_table,
                                                                            on='mutation')


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    df_crashes_obes = extract_crashes_obes_data(sys.argv[1])
    org_model_data = extract_original_model_data(sys.argv[1])

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df_crashes_obes)
    df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list)
    table_mutants_killed_for_no_crashes_obes = compute_and_merge_metric_mutant_tables(df_no_crashes_obes,
                                                                                      org_model_data)
    print(
        'mutants killed by metrics : Mean(LP), Std(Speed), Std(SA) from mutant list not having crashes or obes_________')
    pprint(filter_killed_mutants(table_mutants_killed_for_no_crashes_obes))

    some_crashes_obes_list = build_mutant_list_having_crashes_obes_on_some_models(df_crashes_obes)
    df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                some_crashes_obes_list['mutation'].tolist())
    table_mutants_killed_for_some_crashes_obes = compute_and_merge_metric_mutant_tables(df_some_crashes_obes,
                                                                                        org_model_data)
    print(
        'mutants killed by metrics : Mean(LP), Std(Speed), Std(SA) from mutant list having some crashes or obes_________')
    pprint(filter_killed_mutants(table_mutants_killed_for_some_crashes_obes))
