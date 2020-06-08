# multi-objective-impact

This repository houses the code for reproducing the experimental results from the paper: 

**Esther Rolf, Max Simchowitz, Sarah Dean, Lydia T. Liu, Daniel Björkegren, Moritz Hardt, and Joshua Blumenstock**. [_Balancing Competing Objectives with Noisy Data: Score-Based Classifiers for Welfare-Aware Machine Learning_](https://arxiv.org/abs/2003.06740). Proceedings of the 37th International Conference on Machine Learning (ICML). 2020.


## Instructions
Populate the data folder with the abalone dataset and the credit score data from the links below. Once done, the structure should look like:


multi-objective-impact  
```text
multi-objective-impact  
├── code  
├── data  
│   ├── abalone  
│   │   └── abalone.data  
│   ├── fico  
│   │   ├── totals.csv  	
│   │   ├── transrisk_cdf_by_race_ssa.csv	  
│   │   └── transrisk_performance_by_race_ssa.csv  
```

## Datasets
The outside datasets to reproduce our results are:
* The dataset for the abalone experiments is available for download at [https://archive.ics.uci.edu/ml/datasets/Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone).
* The dataset for the credit score example is available for download at [https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore/data](https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore/data).
* The dataset an intermediate result from [_A Longitudinal Analysis of YouTube's promotion of Conspiracy Videos_](https://www.nytimes.com/interactive/2020/03/02/technology/youtube-conspiracy-theory.html) by authors Marc Fadoul, Guillaume Chaslot and Hany Farid.  (we used it with explicit permission from the authors). For the most up-to-date data from that study, please see [https://github.com/youtube-dataset/conspiracy](https://github.com/youtube-dataset/conspiracy). 


## Code dependencies
We recommend using this code with a 3.7.1 standard Anaconda install. The following packages are required to run our code (we tested our code with the versions in parenthesis):
```text
numpy 1.15.4  
pandas 0.23.4  
sklearn 0.20.1  
matplotlib 3.0.2  
seaborn 0.9.0
```
