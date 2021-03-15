import os
import sys
from decimal import Decimal
from functools import reduce

import numpy as np
import pandas as pd

from analyse_crashes_obes import extract_crashes_obes_data, build_mutant_list_not_having_crashes_obes, \
    build_mutant_list_having_crashes_obes_on_some_models
from extract_model_level_data import extract_data_from_csv
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

cc = [(STD_SPEED, 0.22), (MEAN_LP, -0.75), (STD_SA, -0.67), (MAX_LP, -0.70), (MAX_ACC, 0.32), (MAX_SA, -0.55),
      (Mean_SA, -0.55), (Mean_SAS, 0.60), (Std_SAS, -0.32), (Mean_LS, -0.44), (Std_LS, -0.40), (Min_LP, -0.24),
      (STD_LP, -0.60), (Max_Speed, 0.23), (Mean_Acc, 0.64), (Min_Acc, 0.53), (Std_Acc, -0.22), (Mean_TPP, 0.43),
      (Std_TPP, -0.04), (Min_Speed, -0.12), (Mean_Brake, -0.40), (Count_Braking, -0.51), (STD_BRAKE, -0.44)]


# given list of mutants with no crashes or obes ,
#  it extracts the metrics mean_LP,std_sa,std_speed of those mutant models to a dataframe
def extract_data_based_given_mutant_list(path, no_crashes_obes_list):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if d.startswith(tuple(no_crashes_obes_list))]
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


# this function is to filter out mutants not having 28 sectors
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
        effect_size = ([item for item in cc if item[0] == metric][0][1])
        for i in range(0, 27):
            if remove_this_func_later(mutant_list, i):
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


def compute_and_merge_metric_mutant_tables(df, org_model_data):
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


def map_system_to_model_level_mutant_name(str):
    modified_str = ''
    if any(substring in str for substring in
           ['add_weights_regularisation', 'change_activation_function', 'change_weights_initialisation',
            'remove_activation_function', 'remove_bias_mutated0_MP']):
        modified_str = '_'.join(str.split('_')[:-1])
    if 'delete_training_data' in str:
        words = str.split('_')
        modified_str = '_'.join(['delete_td_mutated0_MP', words[5]])
    if 'output_classes_overlap' in str:
        parameter = '{:f}'.format(Decimal(str[5:].split('_')[5]).normalize())
        mutant_name = '_'.join(str[5:].split('_')[:-1])
        modified_str = '_'.join([mutant_name, parameter])
    if 'change_dropout_rate' in str:
        modified_str = ('_'.join(str.split('_')[:-2]))
    if 'unbalance_train_data' in str:
        words = str.split('_')
        modified_str = ('_'.join(['unbalance_td_mutated0_MP', words[5]]))
    if 'change_label_mutated' in str:
        modified_str = ('_'.join(str.split('_')[:-1]) + '_' + '{:f}'.format(Decimal(str.split('_')[4]).normalize()))
    return modified_str


def make_table_of_mutants_and_metrics_killed_them(lst_of_mutant_names_to_modify, df, model_level_data):
    main_df = []
    for i in lst_of_mutant_names_to_modify:
        data = {}
        tmp_df = df.loc[df['Mutant'] == i]
        if not tmp_df.empty:
            data['Mutant'] = i
            data['killed atleast by one metric'] = 'Yes'
            data['Metrics killed it'] = tmp_df.iloc[0]['Metrics killed']
            lst_of_mutant_names_to_modify = ['add_weights_regularisation', 'change_activation_function',
                                             'change_weights_initialisation',
                                             'remove_activation_function', 'remove_bias_mutated0_MP',
                                             'delete_training_data',
                                             'output_classes_overlap',
                                             'change_dropout_rate', 'unbalance_train_data', 'change_label_mutated']
            if any(ext in i[8:] for ext in lst_of_mutant_names_to_modify):
                modified_mutant_name = map_system_to_model_level_mutant_name(i[8:])
            else:
                modified_mutant_name = i[8:]
            if modified_mutant_name in [j[0] for j in model_level_data]:
                data['Killed by model-level data'] = \
                    [item for item in model_level_data if item[0] == modified_mutant_name][0][1]
            else:
                data['Killed by model-level data'] = 'Missing model-level data'

        else:
            data['Mutant'] = i
            data['killed atleast by one metric'] = 'No'
            data['Metrics killed it'] = []
            if i[8:] in [j[0] for j in model_level_data]:
                data['Killed by model-level data'] = [item for item in model_level_data if item[0] == i[8:]][0][1]
            else:
                data['Killed by model-level data'] = 'Missing model-level data'

        main_df.append(data)

    return pd.DataFrame.from_dict(main_df)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise FileNotFoundError(
            "Insert the correct path to the (1) udacity data directory and (2) model-level data directory")

    df_crashes_obes = extract_crashes_obes_data(sys.argv[1])
    org_model_data = extract_original_model_data(sys.argv[1])

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df_crashes_obes)
    df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list)
    table_mutants_for_no_crashes_obes = compute_and_merge_metric_mutant_tables(df_no_crashes_obes,
                                                                               org_model_data)
    print(
        'computed mutants killed by metrics from mutant list not having crashes or obes_________')

    killed_mutants_for_no_crashes_obes = filter_killed_mutants(table_mutants_for_no_crashes_obes)
    killed_mutants_for_no_crashes_obes.to_csv('killed_info_of_mutants_for_no_crashes_obes.csv')
    metrics_info_for_no_crashes_obes = get_metrics_info(killed_mutants_for_no_crashes_obes)

    some_crashes_obes_list = build_mutant_list_having_crashes_obes_on_some_models(df_crashes_obes)
    df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                some_crashes_obes_list['mutation'].tolist())
    table_mutants_for_some_crashes_obes = compute_and_merge_metric_mutant_tables(df_some_crashes_obes,
                                                                                 org_model_data)
    print(
        '\ncomputed mutants killed by metrics from mutant list having some crashes or obes_________')
    killed_mutants_for_some_crashes_obes = filter_killed_mutants(table_mutants_for_some_crashes_obes)
    killed_mutants_for_some_crashes_obes.to_csv('killed_info_of_mutants_for_some_crashes_obes.csv')
    metrics_info_for_some_crashes_obes = get_metrics_info(killed_mutants_for_some_crashes_obes)
    merged_metrics_info_of_some_and_no = (
        pd.merge(metrics_info_for_no_crashes_obes, metrics_info_for_some_crashes_obes, on=['Metric']).set_index(
            ['Metric']).sum(axis=1).reset_index(name='#mutants killed').sort_values(by='#mutants killed',
                                                                                    ascending=False))
    merged_metrics_info_of_some_and_no.to_csv('metrics_info_on_how_many_mutants_they_kill.csv')
    df_a = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_no_crashes_obes)
    df_b = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_some_crashes_obes)
    model_level_data = extract_data_from_csv(sys.argv[2])

    make_table_of_mutants_and_metrics_killed_them(no_crashes_obes_list, df_a, model_level_data).to_csv(
        'Comparison_with_model_level_of_mutants_for_no_crashes_obes.csv')
    make_table_of_mutants_and_metrics_killed_them(some_crashes_obes_list['mutation'].tolist(), df_b,
                                                  model_level_data).to_csv(
        'Comparison_with_model_level_of_mutants_for_some_crashes_obes.csv')
