import os
import sys
from glob import glob

dir = sys.argv[1]

filenames = []
for root, dirs, files in os.walk(dir + "/"):
    dirs[:] = [d for d in dirs if not d.startswith('udacity_original')]
    for file in files:
        if file.endswith("driving_log_output.csv"):
            filenames.append(os.path.join(root, file))


