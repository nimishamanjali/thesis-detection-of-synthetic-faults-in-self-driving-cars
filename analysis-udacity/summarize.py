import os
import sys
from pprint import pprint

import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)


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
    with open('results(csv)/summarized_results.txt', 'wt') as out:
        all_mutants = get_all_mutants(sys.argv[1])
        out.write("Total mutants: " + str(len(all_mutants))+'\n\n')
        mutants_without_20_runs=pd.read_csv('results(csv)/mutants_lacking_20_models.csv')['mutants'].tolist()
        mutants_without_28_sectors=pd.read_csv('results(csv)/mutants_lacking_28_sectors.csv')['mutants'].tolist()
        mutants_analysed = [x for x in all_mutants if x not in (mutants_without_20_runs + mutants_without_28_sectors)]
        out.write('Mutants analysed(not considering mutants which miss 20 runs and 28 sectors): '+ str(len(mutants_analysed))+'\n')
        pprint(mutants_analysed,stream=out)





