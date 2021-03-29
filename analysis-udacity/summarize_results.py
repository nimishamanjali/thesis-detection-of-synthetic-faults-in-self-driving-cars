import os
import sys
from pprint import pprint

import pandas as pd

from extract_killed_mutants import get_all_metrics

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

metrics = get_all_metrics();


def get_all_mutants(path):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if not d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    for i in filenames:
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))

    df = pd.DataFrame(
        {'mutation': filename})

    return df.groupby(by=['mutation']).size().reset_index()['mutation'].tolist()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise FileNotFoundError("Insert the correct path to the model-level data directory")
    with open('summary.txt', 'wt') as out:
        all_mutants = get_all_mutants(sys.argv[1])
        out.write("Total mutants: " + str(len(all_mutants)) + '\n\n')
        mutants_without_20_runs = pd.read_csv('deepcrime_results(csv)/mutants_lacking_20_models.csv')['mutants'].tolist()
        mutants_without_28_sectors = pd.read_csv('deepcrime_results(csv)/mutants_lacking_28_sectors.csv')['mutants'].tolist()
        mutants_analysed = [x for x in all_mutants if x not in (mutants_without_20_runs + mutants_without_28_sectors)]
        out.write('Mutants analysed(not considering mutants which miss 20 runs and 28 sectors): ' + str(
            len(mutants_analysed)) + '\n')
        pprint(mutants_analysed, stream=out)
        out.write('\n')

        mutants_having_crashes_obes_on_all_model = \
            pd.read_csv('deepcrime_results(csv)/mutant_list_having_crashes_or_obes_on_all_models.csv')['mutation'].tolist()

        out.write(
            'Mutants killed already at level 1 i.e List of mutants that have crashes or OBEs on all 20 model thus definitely has introduced a faulty behaviour: ' + str(
                len(mutants_having_crashes_obes_on_all_model)) + ' mutants')
        out.write('\n')

        pprint(mutants_having_crashes_obes_on_all_model,
               stream=out)
        out.write('\n')
        out.write(
            'analysing further on remaining mutants i.e mutants having crashes or OBEs on some models,mutants not having crashes or OBEs: ' + str(
                len(mutants_analysed) - len(mutants_having_crashes_obes_on_all_model)))
        mutants_for_stat_killing = [x for x in mutants_analysed if x not in mutants_having_crashes_obes_on_all_model]
        mutants_killed_by_stat_killing = []
        df_mutants_killed_info = pd.concat([pd.read_csv(
            'deepcrime_results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv'),
                                            pd.read_csv(
                                                'deepcrime_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')])
        for index, row in df_mutants_killed_info.iterrows():
            for i in metrics:
                if 'killed: True' in row[i]:
                    mutants_killed_by_stat_killing.append(row['mutation'])
                    break;

        out.write('\n')
        out.write('mutants killed on statistical killing approach: ' + str(len(mutants_killed_by_stat_killing)))
        out.write('\n')

        mutants_not_killed_by_stat = len(mutants_for_stat_killing) - len(mutants_killed_by_stat_killing)

        if mutants_not_killed_by_stat >= 0:
            out.write('mutants not killed on statistical killing approach: ' + str(mutants_not_killed_by_stat
                                                                                   ))

        if (len(mutants_for_stat_killing) - len(mutants_killed_by_stat_killing)) != 0:
            pprint([x for x in mutants_for_stat_killing if x not in mutants_killed_by_stat_killing], stream=out)

        df_comparison_model_system = pd.concat(
            [pd.read_csv('deepcrime_results(csv)/Comparison_with_model_level_of_mutants_for_no_crashes_obes.csv'),
             pd.read_csv('deepcrime_results(csv)/Comparison_with_model_level_of_mutants_for_some_crashes_obes.csv')])

        killed_on_both_model = (len(df_comparison_model_system.loc[
                                        (df_comparison_model_system['killed atleast by one metric'] == 'Yes') & (
                                                df_comparison_model_system['Killed by model-level data'] == True)]))

        not_killed_on_both_model = (len(df_comparison_model_system.loc[
                                            (df_comparison_model_system['killed atleast by one metric'] == 'No') & (
                                                    df_comparison_model_system[
                                                        'Killed by model-level data'] == False)]))
        out.write('\n')
        out.write('\n')
        out.write('Information on Comparison between model-level and system-level killing')
        out.write('\n')
        out.write('% : mutant is killed on both model level and system level killing: ' +
                  ("{:.2f}".format(killed_on_both_model / len(df_comparison_model_system) * 100)))
        out.write('\n')
        out.write('% : mutant is not killed on both model level and system level killing: ' +
                  (str(not_killed_on_both_model / len(df_comparison_model_system) * 100)))

        metrics_data = pd.read_csv('deepcrime_results(csv)/metrics_info_on_how_many_mutants_they_kill.csv')
        highest_kill = metrics_data['Metric'].iloc[0]
        metrics_rank_data = \
            pd.read_csv('deepcrime_results(csv)/ranking_of_metrics_based_on_uniquity_of_killing.csv')['Metrics'].iloc[0]

        out.write('\n')
        out.write('\n')
        out.write('Information on metrics')
        out.write('\n')
        out.write('Metrics which kills high number of mutants: ' + str(
            metrics_data.loc[metrics_data['Metric'] == highest_kill]['Metric'].tolist()))
        out.write('\n')
        out.write('The rank 1 metrics which kills mutations when all the other metrics can not kill it: ' + str(
            metrics_rank_data))
        out.write('\n')
        out.write('\n')

        model_level_data_killing_percent = len(
            df_comparison_model_system[df_comparison_model_system['Killed by model-level data'] == True]) / len(
            mutants_analysed)
        crash_obes_killing = len(mutants_having_crashes_obes_on_all_model) / len(mutants_analysed)

        stat_killing = len(mutants_killed_by_stat_killing) / len(mutants_analysed)

        out.write('killing ability percentages: ')
        out.write('\n')
        out.write('Model level killing(on remaining mutants not killed from Crashes/OBEs): ' + "{:.2f}".format(
            model_level_data_killing_percent))
        out.write('\n')
        out.write('Killed if crashes/OBEs on all model: ' + "{:.2f}".format(crash_obes_killing))
        out.write('\n')
        out.write(
            'Killed by statistical killing definition(on remaining mutants not killed from Crashes/OBEs): ' + "{:.2f}".format(
                stat_killing))
