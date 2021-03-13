import os
import sys

import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)

CRASHES_COUNT = '#crashes'
OBEs_COUNT = '#OBEs'


# Provide a list of mutants which do not have any crashes or OBEs
def build_mutant_list_not_having_crashes_obes(whole_df):
    df = whole_df.groupby(by=['mutation'])[[CRASHES_COUNT, OBEs_COUNT]].sum().reset_index()
    df['Neither/Nor'] = np.where((df[CRASHES_COUNT] > 0) | (df[OBEs_COUNT] > 0), "False", "True")
    return df.loc[df['Neither/Nor'] == 'True', 'mutation'].reset_index(drop=True)


# Provide a list of mutants which have crashes or OBEs in some of 20 models and report count in how many models they are
def build_mutant_list_having_crashes_obes_on_some_models(whole_df):
    df_crashes = (whole_df[(whole_df[CRASHES_COUNT] > 0)]).groupby('mutation')[[CRASHES_COUNT]].count().reset_index()
    df_obes = (whole_df[(whole_df[OBEs_COUNT] > 0)]).groupby('mutation')[[OBEs_COUNT]].count().reset_index()
    merged_df = (pd.merge(df_crashes, df_obes, on='mutation', how='outer')).fillna(0)

    return merged_df.astype({CRASHES_COUNT: 'int64', OBEs_COUNT: 'int64'})


# check if crashes or obes count for all 20 models are greater than zero
def check_count_for_all_20_models(x):
    val = all(np.where((x[CRASHES_COUNT] > 0) | (x[OBEs_COUNT] > 0), True, False))
    return str(val)


# Provide a list of mutants which have crashes or OBEs on all 20 models
def build_mutant_list_having_crashes_obes_on_all_models(whole_df):
    df = whole_df.groupby(by=['mutation']).apply(lambda x: check_count_for_all_20_models(x)).reset_index(name='result')
    return df.loc[df['result'] == 'True', 'mutation'].reset_index(drop=True).to_frame()


# returns a df consisting crashes and obes data for all mutant models
def extract_crashes_obes_data(path):
    filenames = []
    for root, dirs, files in os.walk(path + "/"):
        dirs[:] = [d for d in dirs if not d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    filename = []
    crashes = []
    obes = []
    for i in filenames:
        filtered_csv = pd.read_csv(i, usecols=['Crashes', 'OBEs'])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        crashes.append((filtered_csv.sum(axis=0)['Crashes']))
        obes.append((filtered_csv.sum(axis=0)['OBEs']))
        # print(os.path.splitext(i.split('/')[-2])[0])

    return pd.DataFrame(
        {'mutation': filename,
         '#crashes': crashes,
         '#OBEs': obes
         })


def filter_some_from_all(df1, df2):
    return df2[~df2['mutation'].isin(df1['mutation'].tolist())].reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the udacity data directory")

    df = extract_crashes_obes_data(sys.argv[1])
    build_mutant_list_not_having_crashes_obes(df).to_csv('mutant_list_not_having_crashes_obes.csv')

    mutant_lst_with_crashes_or_obes_on_all_model = build_mutant_list_having_crashes_obes_on_all_models(
        df)  # mutants killed
    mutant_lst_with_crashes_or_obes_on_all_model.to_csv('mutant_list_having_crashes_or_obes_on_all_models.csv')

    mutant_lst_with_crashes_or_obes_on_some_and_all_model = build_mutant_list_having_crashes_obes_on_some_models(df)
    filter_some_from_all(mutant_lst_with_crashes_or_obes_on_all_model, mutant_lst_with_crashes_or_obes_on_some_and_all_model).to_csv('mutant_list_having_crashes_or_obes_on_some_models.csv')
