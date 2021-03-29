import sys

import pandas as pd

from analyse_crashes_obes import extract_crashes_obes_data, extract_mutants_without_20_instances, \
    lst_of_mutants_lacking_all_28_sectors, build_mutant_list_not_having_crashes_obes, \
    build_mutant_list_having_crashes_obes_on_all_models, filter_some_from_all, \
    build_mutant_list_having_crashes_obes_on_some_models
from extract_killed_mutants import extract_data_based_given_mutant_list, compute_and_merge_killed_info_mutant_tables, \
    extract_original_model_data, filter_killed_mutants, get_metrics_info, \
    extract_minimal_metrics_that_kills_all_mutants, rank_metrics_in_terms_of_uniquity_of_killing, \
    extract_all_metrics_that_kills_a_mutant


def compare(lst_of_mutant_names_to_modify, df, model_level_data):
    main_df = []

    for i in lst_of_mutant_names_to_modify:
        data = {}
        tmp_df = df.loc[df['Mutant'] == i]
        if not tmp_df.empty:
            data['Mutant'] = i
            data['killed atleast by one metric'] = 'Yes'
            data['Metrics killed it'] = tmp_df.iloc[0]['Metrics killed']
            a = model_level_data.loc[model_level_data['Prefix'] == i + '_1']
            if not a.empty:
                data['Killed by model-level data'] = a['Killed'].iloc[0]
            else:
                data['Killed by model-level data'] = 'Missing model-level data'


        else:
            data['Mutant'] = i
            data['killed atleast by one metric'] = 'No'
            data['Metrics killed it'] = []
            a = model_level_data.loc[model_level_data['Prefix'] == i + '_1']
            if not a.empty:

                data['Killed by model-level data'] = a['Killed'].iloc[0]
            else:
                data['Killed by model-level data'] = 'Missing model-level data'

        main_df.append(data)

    df_result = pd.DataFrame.from_dict(main_df)
    return df_result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise FileNotFoundError(
            "Insert the correct path to the udacity data directory")

    df = extract_crashes_obes_data(sys.argv[1], 'deepmutation')
    print(len(df))
    lst_of_mutants_lacking_20_runs = extract_mutants_without_20_instances(df)
    pd.DataFrame(lst_of_mutants_lacking_20_runs, columns=['mutants']).to_csv(
        'deepmutation_results(csv)/mutants_lacking_20_models.csv')
    pd.DataFrame(list(set(lst_of_mutants_lacking_all_28_sectors)), columns=['mutants']).to_csv(
        'deepmutation_results(csv)/mutants_lacking_28_sectors.csv')

    df = df[~df['mutation'].isin(
        lst_of_mutants_lacking_20_runs + list(set(lst_of_mutants_lacking_all_28_sectors)))].reset_index(drop=True)

    no_crashes_obes_list = build_mutant_list_not_having_crashes_obes(df)
    no_crashes_obes_list.to_csv('deepmutation_results(csv)/mutant_list_not_having_crashes_obes.csv')

    mutant_lst_with_crashes_or_obes_on_all_model = build_mutant_list_having_crashes_obes_on_all_models(
        df)  # mutants killed
    mutant_lst_with_crashes_or_obes_on_all_model.to_csv(
        'deepmutation_results(csv)/mutant_list_having_crashes_or_obes_on_all_models.csv')
    mutant_lst_with_crashes_or_obes_on_some_and_all_model = build_mutant_list_having_crashes_obes_on_some_models(df)

    some_crashes_obes_list = filter_some_from_all(mutant_lst_with_crashes_or_obes_on_all_model,
                                                  mutant_lst_with_crashes_or_obes_on_some_and_all_model)
    some_crashes_obes_list.to_csv(
        'deepmutation_results(csv)/mutant_list_having_crashes_or_obes_on_some_models.csv')

    ############
    org_model_data = extract_original_model_data(sys.argv[2])
    metrics_info_for_no_crashes_obes = pd.DataFrame()
    killed_mutants_for_no_crashes_obes = pd.DataFrame()
    if not no_crashes_obes_list.empty:
        df_no_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1], no_crashes_obes_list)
        table_killed_info_of_mutants_for_no_crashes_obes = compute_and_merge_killed_info_mutant_tables(
            df_no_crashes_obes,
            org_model_data)
        print(
            'computed mutants killed by metrics from mutant list not having crashes or obes_________')

        killed_mutants_for_no_crashes_obes = filter_killed_mutants(table_killed_info_of_mutants_for_no_crashes_obes)
        killed_mutants_for_no_crashes_obes.to_csv('deepmutation_results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv')

        metrics_info_for_no_crashes_obes = get_metrics_info(killed_mutants_for_no_crashes_obes)
    ###

    df_some_crashes_obes = extract_data_based_given_mutant_list(sys.argv[1],
                                                                some_crashes_obes_list['mutation'].tolist(),
                                                                'deepmutation')

    table_killed_info_of_mutants_for_some_crashes_obes = compute_and_merge_killed_info_mutant_tables(
        df_some_crashes_obes,
        org_model_data)

    print(
        '\ncomputed mutants killed by metrics from mutant list having some crashes or obes_________')
    killed_mutants_for_some_crashes_obes = filter_killed_mutants(table_killed_info_of_mutants_for_some_crashes_obes)
    killed_mutants_for_some_crashes_obes.to_csv('deepmutation_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')
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
    extract_minimal_metrics_that_kills_all_mutants(killed_mutants_for_no_crashes_obes,
                                                   killed_mutants_for_some_crashes_obes).to_csv(
        'deepmutation_results(csv)/minimal_set_of_metrics_that_kills_all_mutants')

    rank_metrics_in_terms_of_uniquity_of_killing(killed_mutants_for_no_crashes_obes,
                                                 killed_mutants_for_some_crashes_obes).to_csv(
        'deepmutation_results(csv)/ranking_of_metrics_based_on_uniquity_of_killing.csv')
    merged_metrics_info_of_some_and_no_crashes_obes.reset_index(drop=True).to_csv(
        'deepmutation_results(csv)/metrics_info_on_how_many_mutants_they_kill.csv')

    model_level_data = pd.read_csv(sys.argv[3], usecols=['Prefix', 'Killed'])
    if not killed_mutants_for_no_crashes_obes.empty:
        df_a = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_no_crashes_obes)
        compare(no_crashes_obes_list, df_a, model_level_data, ).to_csv(
            'deepmutation_results(csv)/Comparison_with_model_level_of_mutants_for_no_crashes_obes.csv')

    df_b = extract_all_metrics_that_kills_a_mutant(killed_mutants_for_some_crashes_obes)
    compare(some_crashes_obes_list['mutation'].tolist(), df_b,
            model_level_data).to_csv(
        'deepmutation_results(csv)/Comparison_with_model_level_of_mutants_for_some_crashes_obes.csv')
