import os
import sys

import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', -1)


def get_list_of_mutants_without_any_crashes_or_OBEs(whole):
    # 3rd point-Provide a list of mutants which do not have any crashes or OBEs
    df = whole.groupby(by=['mutation'])[['#crashes', '#OBEs']].sum().reset_index()
    df["Neither"] = np.where((df["#crashes"] > 0) | (df["#OBEs"] > 0), "False", "True")
    return df.loc[df['Neither'] == 'True', 'mutation'].tolist()


def get_list_of_some(whole):
    # Provide a list of mutants which have crashes or OBEs in some of 20 models and report in how many models they are
    one = (whole[(whole['#crashes'] > 0)]).groupby('mutation')[['#crashes']].count().reset_index()
    two = (whole[(whole['#OBEs'] > 0)]).groupby('mutation')[['#OBEs']].count().reset_index()
    df_2 = (pd.merge(one, two, on='mutation', how='outer')).fillna(0)
    return df_2


def f(x):
    val = (all(np.where((x['#crashes'] > 0) | (x['#OBEs'] > 0), True, False)))
    return str(val)


def get_list_of_all():
    # 1st point - Provide a list of mutants which have crashes or OBEs on all 20 models
    df = whole.groupby(by=['mutation']).apply(lambda x: f(x)).reset_index(name='result')
    return df.loc[df['result'] == 'True', 'mutation'].tolist()


if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise FileNotFoundError("Insert the correct path to the tensorflow directory")
    dir = sys.argv[1]
    filenames = []
    for root, dirs, files in os.walk(dir + "/"):
        dirs[:] = [d for d in dirs if not d.startswith('udacity_original')]
        for file in files:
            if file.endswith("driving_log_output.csv"):
                filenames.append(os.path.join(root, file))

    print(len(filenames))
    filename = []
    crashes = []
    obes = []
    for i in filenames:
        my_filtered_csv = pd.read_csv(i, usecols=['Crashes', 'OBEs'])
        filename.append('_'.join((os.path.splitext(i.split('/')[-2])[0]).split('_')[:-1]))
        crashes.append((my_filtered_csv.sum(axis=0)['Crashes']))
        obes.append((my_filtered_csv.sum(axis=0)['OBEs']))
        # print(os.path.splitext(i.split('/')[-2])[0])

    whole = pd.DataFrame(
        {'mutation': filename,
         '#crashes': crashes,
         '#OBEs': obes
         })

    print(whole)
    # print(get_list_of_mutants_without_any_crashes_or_OBEs(whole))
    # print(get_list_of_all())
    # print(get_list_of_some(whole))
