import os
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
# metrics = [MEAN_LP, STD_SPEED, STD_SA, MAX_LP, MAX_ACC, Std_LS]

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


def extract_metric_based_on_level_of_killing(row, i):
    if any(e in row['killed'] for e in i):
        return row['Metric']


def extract_metrics_based_on_killing_level_of_mutants(x, df_info):
    val = (df_info.apply(
        lambda row: extract_metric_based_on_level_of_killing(row, x['Metric'].tolist()), axis=1).tolist())
    return [x for x in val if x is not None]


def rank_metrics_in_terms_of_uniquity_of_killing(killed_mutants_for_no_crashes_obes,
                                                 killed_mutants_for_some_crashes_obes):
    whole_df = pd.concat([killed_mutants_for_no_crashes_obes, killed_mutants_for_some_crashes_obes]).reset_index(
        drop=True)

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
            ranks.append({'ranking': count, 'metrics ': metric_lst})
            all_metrics = [x for x in all_metrics if x not in row['metrics at level']]
            count += 1

    return pd.DataFrame.from_dict(ranks).reset_index(drop=True)


def extract_minimal_metrics_that_kills_all_mutants(table_mutants_for_no_crashes_obes,
                                                   killed_mutants_for_some_crashes_obes):
    whole_df = pd.concat([table_mutants_for_no_crashes_obes, killed_mutants_for_some_crashes_obes]).reset_index(
        drop=True)
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


def map_mutants_with_binary_search(str):
    modified_str = ''
    if any(substring in str for substring in
           ['change_epochs', 'change_learning_rate']):
        modified_str = str
    if 'delete_training_data' in str:
        words = str.split('_')
        modified_str = '_'.join(['delete_td_mutated0_MP', words[5]])
    if 'output_classes_overlap' in str:
        parameter = '{:f}'.format(Decimal(str[5:].split('_')[5]).normalize())
        mutant_name = '_'.join(str[5:].split('_')[:-1])
        modified_str = '_'.join([mutant_name, parameter])
    if 'unbalance_train_data' in str:
        words = str.split('_')
        modified_str = ('_'.join(['unbalance_td_mutated0_MP', words[5]]))
    if 'change_label_mutated' in str:
        modified_str = ('_'.join(str.split('_')[:-1]) + '_' + '{:f}'.format(Decimal(str.split('_')[4]).normalize()))
    return modified_str


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


def check_if_mutant_is_killed_based_on_range(mutant, range_data, org):
    match = []
    for i, j in range_data.iterrows():
        if any(mutant in string for string in j['range']):
            match.append((j['range'], j['killed']))
    if match:
        val = ''

        for i in match:
            if float(i[0][0].split('_')[-1]) <= float(org.split('_')[-1]) <= float(i[0][1].split('_')[-1]):
                val = i[1]
                break

            elif len(i[0]) == 1:
                if (int(i[0][0].split('_')[-1])) == 100:
                    val = i[1]
            else:
                val = 'Missing model-level data'
        return val
    else:
        return ''


def make_table_of_mutants_and_metrics_killed_them(lst_of_mutant_names_to_modify, df, model_level_data, range_data):
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

    df = pd.DataFrame.from_dict(main_df)

    for index, row in df.iterrows():
        if row['Killed by model-level data'] == 'Missing model-level data':
            mutant_between_range = ('_'.join(map_mutants_with_binary_search(row['Mutant'][8:]).split('_')[:-1]))
            row['Killed by model-level data'] = (
                check_if_mutant_is_killed_based_on_range(mutant_between_range, range_data, row['Mutant']))
    return df


def build_range_binary_search(lst):
    res = [[lst[i], lst[i + 1]]
           for i in range(len(lst) - 1)]
    if not res:
        return [lst]
    return res


def construct_range_table_for_binary_search(model_level_data, bs):
    main_df = []
    for i in bs:

        data = []
        for j in [x for x in model_level_data]:

            if i in j[0]:
                data.append(j)
        sorted_data = sorted(data, key=lambda x: float(x[0].split('_')[-1]))

        for j in build_range_binary_search(sorted_data):
            if len(j) > 1:
                if j[0][1] == j[1][1]:
                    main_df.append({'range': [j[0][0], j[1][0]], 'killed': j[0][1]})
            else:
                main_df.append({'range': [j[0][0]], 'killed': j[0][1]})
    return pd.DataFrame.from_dict(main_df)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise FileNotFoundError(
            "Insert the correct path to the (1) udacity data directory and (2) model-level data directory")

    df_crashes_obes = extract_crashes_obes_data(sys.argv[1])
    org_model_data = extract_original_model_data(sys.argv[1])
    model_level_data = extract_data_from_csv(sys.argv[2])
    mutants_with_range = ['delete_td', 'output_classes', 'change_label', 'change_epochs', 'change_learning_rate',
                          'unbalance_td']

    range_data = construct_range_table_for_binary_search(model_level_data, mutants_with_range)

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df_crashes_obes)
    df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list)
    table_mutants_for_no_crashes_obes = compute_and_merge_metric_mutant_tables(df_no_crashes_obes,
                                                                               org_model_data)
    print(
        'computed mutants killed by metrics from mutant list not having crashes or obes_________')

    killed_mutants_for_no_crashes_obes = filter_killed_mutants(table_mutants_for_no_crashes_obes)
    killed_mutants_for_no_crashes_obes.to_csv('results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv')

    metrics_info_for_no_crashes_obes = get_metrics_info(killed_mutants_for_no_crashes_obes)

    some_crashes_obes_list = build_mutant_list_having_crashes_obes_on_some_models(df_crashes_obes)
    df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                some_crashes_obes_list['mutation'].tolist())
    table_mutants_for_some_crashes_obes = compute_and_merge_metric_mutant_tables(df_some_crashes_obes,
                                                                                 org_model_data)
    print(
        '\ncomputed mutants killed by metrics from mutant list having some crashes or obes_________')
    killed_mutants_for_some_crashes_obes = filter_killed_mutants(table_mutants_for_some_crashes_obes)
    killed_mutants_for_some_crashes_obes.to_csv('results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')
    metrics_info_for_some_crashes_obes = get_metrics_info(killed_mutants_for_some_crashes_obes)
    merged_metrics_info_of_some_and_no = (
        pd.merge(metrics_info_for_no_crashes_obes, metrics_info_for_some_crashes_obes, on=['Metric']).set_index(
            ['Metric']).sum(axis=1).reset_index(name='#mutants killed').sort_values(by='#mutants killed',
                                                                                    ascending=False))
    extract_minimal_metrics_that_kills_all_mutants(killed_mutants_for_no_crashes_obes,
                                                   killed_mutants_for_some_crashes_obes).to_csv(
        'results(csv)/minimal_set_of_metrics_that_kills_all_mutants')
    rank_metrics_in_terms_of_uniquity_of_killing(killed_mutants_for_no_crashes_obes,
                                                 killed_mutants_for_some_crashes_obes).to_csv(
        'results(csv)/ranking_of_metrics_based_on_uniquity_of_killing.csv')
    merged_metrics_info_of_some_and_no.to_csv('results(csv)/metrics_info_on_how_many_mutants_they_kill.csv')
    df_a = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_no_crashes_obes)
    df_b = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_some_crashes_obes)
    make_table_of_mutants_and_metrics_killed_them(no_crashes_obes_list, df_a, model_level_data, range_data).to_csv(
        'results(csv)/Comparison_with_model_level_of_mutants_for_no_crashes_obes.csv')
    make_table_of_mutants_and_metrics_killed_them(some_crashes_obes_list['mutation'].tolist(), df_b,
                                                  model_level_data, range_data).to_csv(
        'results(csv)/Comparison_with_model_level_of_mutants_for_some_crashes_obes.csv')
