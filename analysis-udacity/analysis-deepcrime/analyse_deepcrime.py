import os
import sys
from decimal import Decimal

import pandas as pd

from utilities import extract_original_model_data, \
    extract_data_based_given_mutant_list, compute_and_merge_killed_info_mutant_tables, filter_killed_mutants, \
    get_metrics_info, extract_minimal_metrics_that_kills_all_mutants, rank_metrics_in_terms_of_uniquity_of_killing, \
    extract_all_metrics_that_kills_a_mutant
from utilities_to_analyse_crashes_obes import extract_crashes_obes_data, build_mutant_list_not_having_crashes_obes, \
    filter_some_from_all, build_mutant_list_having_crashes_obes_on_some_models, extract_mutants_without_20_instances, \
    lst_of_mutants_lacking_all_28_sectors, build_mutant_list_having_crashes_obes_on_all_models


def make_tuple_from_string(str):
    return tuple(str.split(' '))


def remove_if_nan(lst):
    return [i for i in lst if 'nan' not in i]


def extract_model_level_data(path):
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


def compare_with_model_level(lst_of_mutant_names_to_modify, df, model_level_data, range_data):
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


def construct_range_table_for_mutants_with_binary_search(model_level_data, bs):
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


def do_analysis(exclude_mutants_having_crashes_or_obes_on_all_from_some=True):
    df = extract_crashes_obes_data(sys.argv[1], 'deepcrime')

    lst_of_mutants_lacking_20_runs = extract_mutants_without_20_instances(df)
    pd.DataFrame(lst_of_mutants_lacking_20_runs, columns=['mutants']).to_csv(
        'deepcrime_results(csv)/mutants_lacking_20_models.csv')
    pd.DataFrame(list(set(lst_of_mutants_lacking_all_28_sectors)), columns=['mutants']).to_csv(
        'deepcrime_results(csv)/mutants_lacking_28_sectors.csv')

    df = df[~df['mutation'].isin(
        lst_of_mutants_lacking_20_runs + list(set(lst_of_mutants_lacking_all_28_sectors)))].reset_index(drop=True)

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df)
    no_crashes_obes_list.to_csv('deepcrime_results(csv)/mutant_list_not_having_crashes_obes.csv')

    mutant_lst_with_crashes_or_obes_on_all_model = build_mutant_list_having_crashes_obes_on_all_models(
        df)  # mutants killed
    mutant_lst_with_crashes_or_obes_on_all_model.to_csv(
        'deepcrime_results(csv)/mutant_list_having_crashes_or_obes_on_all_models.csv')

    mutant_lst_with_crashes_or_obes_on_some_and_all_model = build_mutant_list_having_crashes_obes_on_some_models(df)

    some_crashes_obes_list = filter_some_from_all(mutant_lst_with_crashes_or_obes_on_all_model,
                                                  mutant_lst_with_crashes_or_obes_on_some_and_all_model)
    some_crashes_obes_list.to_csv(
        'deepcrime_results(csv)/mutant_list_having_crashes_or_obes_on_some_models.csv')
    ##

    mutants_with_range = ['delete_td', 'output_classes', 'change_label', 'change_epochs', 'change_learning_rate',
                          'unbalance_td']

    org_model_data = extract_original_model_data(sys.argv[1])
    model_level_data = extract_model_level_data(sys.argv[2])
    mutants_range_data = construct_range_table_for_mutants_with_binary_search(model_level_data, mutants_with_range)

    # no crashes or obes
    metrics_info_for_no_crashes_obes = pd.DataFrame()
    killed_mutants_for_no_crashes_obes = pd.DataFrame()
    if not no_crashes_obes_list.empty:
        df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list, 'deepcrime')
        table_killed_info_of_mutants_for_no_crashes_obes = compute_and_merge_killed_info_mutant_tables(
            df_no_crashes_obes,
            org_model_data)
        print(
            'computed mutants killed by metrics from mutant list not having crashes or obes_________')

        killed_mutants_for_no_crashes_obes = filter_killed_mutants(table_killed_info_of_mutants_for_no_crashes_obes)
        killed_mutants_for_no_crashes_obes.to_csv(
            'deepcrime_results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv')

        metrics_info_for_no_crashes_obes = get_metrics_info(killed_mutants_for_no_crashes_obes)

    ##some crashes or obes
    if exclude_mutants_having_crashes_or_obes_on_all_from_some is False:
        some_crashes_obes_list = mutant_lst_with_crashes_or_obes_on_some_and_all_model

    metrics_info_for_some_crashes_obes = pd.DataFrame()
    killed_mutants_for_some_crashes_obes = pd.DataFrame()
    if not some_crashes_obes_list.empty:
        df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                    some_crashes_obes_list['mutation'].tolist(),
                                                                    'deepcrime')
        table_killed_info_of_mutants_for_some_crashes_obes = compute_and_merge_killed_info_mutant_tables(
            df_some_crashes_obes,
            org_model_data)

        print(
            '\ncomputed mutants killed by metrics from mutant list having some crashes or obes_________')

        killed_mutants_for_some_crashes_obes = filter_killed_mutants(table_killed_info_of_mutants_for_some_crashes_obes)
        killed_mutants_for_some_crashes_obes.to_csv(
            'deepcrime_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')
        metrics_info_for_some_crashes_obes = get_metrics_info(killed_mutants_for_some_crashes_obes)

    if metrics_info_for_no_crashes_obes.empty:
        merged_metrics_info_of_some_and_no_crashes_obes = metrics_info_for_some_crashes_obes

    elif metrics_info_for_some_crashes_obes.empty:
        merged_metrics_info_of_some_and_no_crashes_obes = metrics_info_for_no_crashes_obes
    else:
        merged_metrics_info_of_some_and_no_crashes_obes = (
            pd.merge(metrics_info_for_no_crashes_obes, metrics_info_for_some_crashes_obes, on=['Metric']).set_index(
                ['Metric']).sum(axis=1).reset_index(name='#mutants killed').sort_values(by='#mutants killed',
                                                                                        ascending=False))
    merged_metrics_info_of_some_and_no_crashes_obes.reset_index(drop=True).to_csv(
        'deepcrime_results(csv)/metrics_info_on_how_many_mutants_they_kill.csv')

    if not killed_mutants_for_no_crashes_obes.empty or killed_mutants_for_some_crashes_obes.empty:
        if killed_mutants_for_no_crashes_obes.empty:
            whole_df = killed_mutants_for_some_crashes_obes
        elif killed_mutants_for_some_crashes_obes.empty:
            whole_df = killed_mutants_for_no_crashes_obes
        else:
            whole_df = pd.concat(
                [killed_mutants_for_no_crashes_obes, killed_mutants_for_some_crashes_obes]).reset_index(
                drop=True)
        extract_minimal_metrics_that_kills_all_mutants(whole_df).to_csv(
            'deepcrime_results(csv)/minimal_set_of_metrics_that_kills_all_mutants')

        rank_metrics_in_terms_of_uniquity_of_killing(whole_df).to_csv(
            'deepcrime_results(csv)/ranking_of_metrics_based_on_uniquity_of_killing.csv')

    if not killed_mutants_for_no_crashes_obes.empty:
        df_a = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_no_crashes_obes)
        compare_with_model_level(no_crashes_obes_list, df_a, model_level_data,
                                 mutants_range_data).to_csv(
            'deepcrime_results(csv)/Comparison_with_model_level_of_mutants_for_no_crashes_obes.csv')

    if not killed_mutants_for_some_crashes_obes.empty:
        df_b = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_some_crashes_obes)

        compare_with_model_level(some_crashes_obes_list['mutation'].tolist(), df_b,
                                 model_level_data, mutants_range_data).to_csv(
            'deepcrime_results(csv)/Comparison_with_model_level_of_mutants_for_some_crashes_obes.csv')


if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise FileNotFoundError(
            "Insert the correct path to the (1) udacity data directory and (2) model-level data directory")
    # provide parameter False if you want to do analysis on all models without excluding the ones with crashes/OBEs on all models
    do_analysis()
