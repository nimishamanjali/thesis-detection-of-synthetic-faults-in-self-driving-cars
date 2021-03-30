import os
import sys
from functools import reduce

import numpy as np
import pandas as pd

from stats import is_diff_sts

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

STD_SPEED = 'Std(Speed)'
MEAN_LP = 'Mean(LP)'
STD_SA = 'Std(SA)'
MAX_LP = 'Max(LP)'
MAX_ACC = 'Max(Acc)'
MAX_SA = 'Max(SA)'
Mean_SA = 'Mean(SA)'
Mean_SAS = 'Mean(SAS)'
Std_SAS = 'Std(SAS)'
Mean_LS = 'Mean(LS)'
Std_LS = 'Std(LS)'
Min_LP = 'Min(LP)'
STD_LP = 'Std(LP)'
Max_Speed = 'Max(Speed)'
Mean_Acc = 'Mean(Acc)'
Min_Acc = 'Min(Acc)'
Std_Acc = 'Std(Acc)'
Mean_TPP = 'Mean(TPP)'
Std_TPP = 'Std(TPP)'
# metrics having problem in glm.fit()
Mean_Brake = 'Mean(Brake)'
Count_Braking = 'Count(Braking)'
STD_BRAKE = 'Std(Brake)'
Min_Speed = 'Min(Speed)'

metrics = [MEAN_LP, STD_SPEED, STD_SA, MAX_LP, MAX_ACC, MAX_SA, Mean_SA, Mean_SAS, Std_SAS, Mean_LS, Std_LS, Min_LP,
           STD_LP, Max_Speed,
           Mean_Acc, Min_Acc, Std_Acc, Mean_TPP, Std_TPP]


# metrics = [MEAN_LP, STD_SPEED, STD_SA, MAX_LP, MAX_ACC, Std_LS]
def get_all_metrics():
    return metrics


cc = [(STD_SPEED, 0.22), (MEAN_LP, -0.75), (STD_SA, -0.67), (MAX_LP, -0.70), (MAX_ACC, 0.32), (MAX_SA, -0.55),
      (Mean_SA, -0.55), (Mean_SAS, 0.60), (Std_SAS, -0.32), (Mean_LS, -0.44), (Std_LS, -0.40), (Min_LP, -0.24),
      (STD_LP, -0.60), (Max_Speed, 0.23), (Mean_Acc, 0.64), (Min_Acc, 0.53), (Std_Acc, -0.22), (Mean_TPP, 0.43),
      (Std_TPP, -0.04), (Min_Speed, -0.12), (Mean_Brake, -0.40), (Count_Braking, -0.51), (STD_BRAKE, -0.44)]


# given list of mutants with no crashes or obes ,
#  it extracts the metrics mean_LP,std_sa,std_speed of those mutant models to a dataframe
def extract_data_based_given_mutant_list(path, no_crashes_obes_list, mutation_tool):
    if mutation_tool == 'deepmutation':
        no_crashes_obes_list = [element + "_1.h5" for element in no_crashes_obes_list]

    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        if mutation_tool == 'deepmutation':
            dirs[:] = [d for d in dirs if d.endswith(tuple(no_crashes_obes_list))]
        if mutation_tool == 'deepcrime':
            dirs[:] = [d for d in dirs if d.startswith(tuple(no_crashes_obes_list))]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    full_list = []
    for i in filenames:
        cols = ['mutation'] + metrics
        tmp_df = pd.DataFrame(columns=cols)
        filtered_csv = pd.read_csv(i, usecols=metrics)

        if mutation_tool == 'deepmutation':
            tmp_df.at[0, 'mutation'] = '_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[1:-1])
        if mutation_tool == 'deepcrime':
            tmp_df.at[0, 'mutation'] = '_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1])
        for j in metrics:
            tmp_df.at[0, j] = np.array(filtered_csv[j].tolist())

        full_list.append(tmp_df)

    main_df = pd.concat(full_list)
    return main_df.reset_index(drop=True)


# extract and save the original model data to a dataframe
def extract_original_model_data(path):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    full_list = []
    for i in filenames:
        cols = ['mutation'] + metrics
        tmp_df = pd.DataFrame(columns=cols)
        filtered_csv = pd.read_csv(i, usecols=metrics)
        tmp_df.at[0, 'mutation'] = '_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1])

        for j in metrics:
            tmp_df.at[0, j] = np.array(filtered_csv[j].tolist())

        full_list.append(tmp_df)

    main_df = pd.concat(full_list)
    return main_df.reset_index(drop=True)


# perform statistical def of killing comparing each 20 mutant version to each 20 original model
def compute_table_for_metric(org, x, metric):
    killed = None
    if len(x[metric]) == 20:

        mutant_list = x[metric].tolist()
        org_list = org[metric].tolist()

        effect_size = ([item for item in cc if item[0] == metric][0][1])
        for i in range(0, 27):
            data = is_diff_sts([item[i] for item in org_list], [item[i] for item in mutant_list], effect_size)
            if data[0]:
                killed = 'killed: ' + str(data[0]) + ',' + 'Sector number: ' + str(i) + ',' + ' p_value: ' + str(
                    data[1]) + ',' + ' effect_size: ' + str(data[2])
                break

    return tuple(killed.split(",")) if killed is not None else 'killed: false'


def filter_killed_mutants(df):
    merged = []
    for i in metrics:
        killed_mutants = df.loc[df[i] != 'killed: false'][['mutation', i]]

        merged.append(killed_mutants)
    return reduce(lambda df1, df2: pd.merge(df1, df2, on='mutation', how='outer'), merged).replace(np.nan,
                                                                                                   'Not killed by this metric',
                                                                                                   regex=True)


def compute_and_merge_killed_info_mutant_tables(df, org_model_data):
    dfs = []
    for i in metrics:
        dfs.append((df.groupby(by=['mutation'])).apply(
            lambda x: compute_table_for_metric(org_model_data, x, i)).reset_index(
            name=i))
    return reduce(lambda left, right: pd.merge(left, right, on=['mutation']),
                  dfs)


def get_metrics_info(table_mutants_for_no_crashes_obes):
    lst = []
    for i in metrics:
        lst.append((i, len(
            table_mutants_for_no_crashes_obes[(table_mutants_for_no_crashes_obes[i] != 'Not killed by this metric')])))
    main_df = []
    for i in lst:
        main_df.append({'Metric': i[0], '#mutants killed': i[1]})

    return pd.DataFrame.from_dict(main_df).sort_values(by='#mutants killed', ascending=False).reset_index(drop=True)


def extract_metric_based_on_level_of_killing(row, i):
    if any(e in row['killed'] for e in i):
        return row['Metric']


def extract_metrics_based_on_killing_level_of_mutants(x, df_info):
    val = (df_info.apply(
        lambda row: extract_metric_based_on_level_of_killing(row, x['Metric'].tolist()), axis=1).tolist())
    return [x for x in val if x is not None]


def rank_metrics_in_terms_of_uniquity_of_killing(whole_df):
    lst = []
    main_df = []
    for i in metrics:
        lst.append((i,
                    whole_df[
                        (whole_df[i] != 'Not killed by this metric')]['mutation'].tolist()))
    for i in lst:
        main_df.append({'Metric': i[0], 'killed': i[1]})
    df_info = pd.DataFrame.from_dict(main_df).reset_index(drop=True)
    flat_list = [item for sublist in df_info['killed'].tolist() for item in sublist]
    counts = []
    for i in set(flat_list):
        i_count = 0
        for index, row in df_info.iterrows():
            i_count += (row['killed'].count(i))
        counts.append({'Metric': i, '#metrics killed': i_count})
    df = pd.DataFrame.from_dict(counts).sort_values('#metrics killed')
    all_metrics = df_info['Metric'].tolist()
    tmp_df = df.groupby(by=['#metrics killed']).apply(
        lambda x: extract_metrics_based_on_killing_level_of_mutants(x, df_info)).reset_index(name='metrics at level')
    ranks = []
    count = 0
    for index, row in tmp_df.iterrows():
        metric_lst = [x for x in all_metrics if x in (row['metrics at level'])]
        if metric_lst:
            ranks.append({'Ranking': count, 'Metrics': metric_lst})
            all_metrics = [x for x in all_metrics if x not in row['metrics at level']]
            count += 1

    return pd.DataFrame.from_dict(ranks).reset_index(drop=True)


def extract_minimal_metrics_that_kills_all_mutants(whole_df):
    lst = []
    main_df = []
    killed_metrics = []
    for i in metrics:
        lst.append((i, len(
            whole_df[(whole_df[i] != 'Not killed by this metric')]),
                    whole_df[
                        (whole_df[i] != 'Not killed by this metric')]['mutation'].tolist()))
    for i in lst:
        main_df.append({'Metric': i[0], '#mutants killed': i[1], 'killed mutants': i[2]})
        killed_metrics.append(i[2])
    df_info = pd.DataFrame.from_dict(main_df).sort_values(by='#mutants killed', ascending=False).reset_index(drop=True)
    flat_list = set([item for sublist in killed_metrics for item in sublist])
    minimal = []
    for index, row in df_info.iterrows():
        flat_list = [x for x in flat_list if x not in row[2]]
        minimal.append((row[0], row[1]))
        if not flat_list:
            break

    df_samples = df_info.loc[df_info['#mutants killed'].isin([x[1] for x in minimal])]
    for index, row in df_samples.iterrows():
        minimal.append((row[0], row[1]))

    return pd.DataFrame([x[0] for x in list(set(minimal))], columns=['metrics'])


def extract_all_metrics_that_kills_a_mutant(killed_mutants_for_no_crashes_obes):
    metrics_killed = ((killed_mutants_for_no_crashes_obes.apply(
        lambda row: (row[row != 'Not killed by this metric'].index, row[0]), axis=1)).to_frame(name='metrics_killed'))
    metric_lst = []
    mutant_lst = []
    for index, row in metrics_killed.iterrows():
        mutant_lst.append(row[0][1])
        metric_lst.append(row[0][0].tolist()[1:])

    return pd.DataFrame({'Mutant': mutant_lst,
                         'Metrics killed': metric_lst
                         })


