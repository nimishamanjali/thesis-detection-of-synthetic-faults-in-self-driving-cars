import pandas as pd

from utilities import metrics, extract_minimal_metrics_that_kills_all_mutants

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

killed_mutants_for_no_crashes_obes = pd.read_csv(
    'analysis-deepcrime/deepcrime_results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv')
killed_mutants_for_some_crashes_obes = pd.read_csv(
    'analysis-deepcrime/deepcrime_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')
killed_sec = pd.read_csv(
    'analysis-deepmutation/deepmutation_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')

whole_df = pd.concat(
    [killed_mutants_for_no_crashes_obes, killed_mutants_for_some_crashes_obes, killed_sec]).reset_index(
    drop=True)

counts = []
for index, row in whole_df.iterrows():
    l = []
    for i in metrics:
        if 'killed: True' in row[i]:
            l.append(i)
    counts.append({'mutant': row['mutation'], '#metrics killed': len(l), 'metrics': l})
df = pd.DataFrame.from_dict(counts).sort_values('#metrics killed').reset_index(drop=True)
df.head(10).to_csv('hardly_killed_mutants.csv')
# minimal set of metrics that kills mutants of both tool together
extract_minimal_metrics_that_kills_all_mutants(whole_df).to_csv(
    'minimal_set_of_killing_metrics_for_both_tools_together.csv')
