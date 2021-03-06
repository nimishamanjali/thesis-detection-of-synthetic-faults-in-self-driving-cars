import os
import sys

import pandas as pd

from analyse_crashes_obes import extract_crashes_obes_data, build_mutant_list_not_having_crashes_obes
from stats import is_diff_sts

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

STD_SPEED = 'Std(Speed)'
MEAN_LP = 'Mean(LP)'
STD_SA = 'Std(SA)'


def extract_data_based_on_no_crashes_obes_list(path, no_crashes_obes_list):
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
        mean_lp.append((filtered_csv.sum(axis=0)[MEAN_LP]))
        std_sa.append((filtered_csv.sum(axis=0)[STD_SA]))
        std_speed.append((filtered_csv.sum(axis=0)[STD_SPEED]))

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed
         })


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
        mean_lp.append((filtered_csv.sum(axis=0)[MEAN_LP]))
        std_sa.append((filtered_csv.sum(axis=0)[STD_SA]))
        std_speed.append((filtered_csv.sum(axis=0)[STD_SPEED]))

    return pd.DataFrame(
        {'mutation': filename,
         MEAN_LP: mean_lp,
         STD_SA: std_sa,
         STD_SPEED: std_speed
         })


def compute_table_for_metric(org, x, metric):
    data = is_diff_sts(x[metric].tolist(), org[metric].tolist())
    return 'killed: ' + str(data[0]) + '; p_value: ' + str(round(data[1], 3)) + '; effect_size: ' + str(
        round(data[2], 3))


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    df_crashes_obes = extract_crashes_obes_data(sys.argv[1])
    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df_crashes_obes)
    df = extract_data_based_on_no_crashes_obes_list(sys.argv[1], no_crashes_obes_list)
    org_model_data = extract_original_model_data(sys.argv[1])
    mean_lp_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, MEAN_LP)).reset_index(
        name=MEAN_LP)
    std_sa_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, STD_SA)).reset_index(
        name=STD_SA)
    std_speed_stat_table = (df.groupby(by=['mutation'])).apply(
        lambda x: compute_table_for_metric(org_model_data, x, STD_SPEED)).reset_index(
        name='STD_SPEED')
    table_meanlp_stdsa_stdspeed = mean_lp_stat_table.merge(std_sa_stat_table, on='mutation').merge(std_speed_stat_table,
                                                                                                   on='mutation')
    print(table_meanlp_stdsa_stdspeed)
