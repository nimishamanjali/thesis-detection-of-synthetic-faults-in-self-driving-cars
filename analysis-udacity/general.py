import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)
from utilities import extract_minimal_metrics_that_kills_all_mutants
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
killed_mutants_for_no_crashes_obes = pd.read_csv(
    'analysis-deepcrime/deepcrime_results(csv)/killed_info_of_mutants_for_no_crashes_obes.csv')
killed_mutants_for_some_crashes_obes = pd.read_csv(
    'analysis-deepcrime/deepcrime_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')
killed_sec = pd.read_csv(
    'analysis-deepmutation/deepmutation_results(csv)/killed_info_of_mutants_for_some_crashes_obes.csv')

whole_df = pd.concat(
    [killed_mutants_for_no_crashes_obes, killed_mutants_for_some_crashes_obes, killed_sec]).reset_index(
    drop=True)
print(len(metrics))

#print(whole_df.to_csv('tmp1.csv'))
counts = []
for index, row in whole_df.iterrows():
    l=[]
    for i in metrics:
        if 'killed: True' in row[i]:
            #print(row['mutation'])
            l.append(i)
    counts.append({'mutant': row['mutation'], '#metrics killed': len(l),'metrics':l})
df = pd.DataFrame.from_dict(counts).sort_values('#metrics killed').reset_index(drop=True)
print(df.to_csv('possible_rq.csv'))
print(whole_df['mutation'])