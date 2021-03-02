import os
import sys
import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
#pd.set_option('display.max_colwidth', -1)
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
    #print(os.path.splitext(i.split('/')[-2])[0])

whole = pd.DataFrame(
    {'mutation': filename,
     '#crashes': crashes,
     '#OBEs': obes
     })
#print(whole)
df=whole.groupby(by=['mutation'])[['#crashes','#OBEs']].sum()
df["Neither"] = np.where((df["#crashes"] >0) | (df["#OBEs"] >0) , "False", "True")
print(df)