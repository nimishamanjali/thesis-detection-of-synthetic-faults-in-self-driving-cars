# thesis-detection-of-synthetic-faults-in-self-driving-cars
Master thesis

## Python Version
#### 3.6.8

## Usage

* To execute the analysis of Deepcrime tool, run program analyse_deepcrime.py with following 2 parameters:
  * (1) path to deepcrime data directory
  * (2) path to model-level data directory

* To execute the analysis of Deepmutation tool, run program analyse_deepmutation.py, with following 3 parameters:
  * (1) path to deepmutation data directory
  * (2) path to original-model data directory
  * (3) path to model-level data csv

* To run the program summarize_results.py, provide following 2 parameter:
  * (1) name of mutation tool you want to analyse (deepmutation/deepcrime)
  * (2) path to data(deepmutation/deepcrime) directory for which you want to summarize results