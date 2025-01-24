# Overview

The scripts presented herein aim to give results files for the Snake Strikes Back competition, hosted by Tristan Allard, PHD, Université de Rennes.

They consist in a few Jupyter Notebooks calling python scripts located in the root folder.

## How to use

To run a whole experience, roughly three parameters need to be configured by the user :

1. In makeClasssifierTrainingDataset2.ipynb, specify the privateDataset_Path leading to the datasets you want your classifiers to be trained on
2. In createClassifiers2.ipynb, uncomment the classifiers you don't want to use in modelsToUse. Please be aware that the last models are very long to be trained if your training sets is large.
3. In classifyTargets2.ipynb, specify the taskNumber _(1 to 4)_

Then simply run classifyTargets2.ipynb and inspect the file create in results/Task X. For reproductibility you can find the training dataset created for the experience and the performances metrics of the classifiers.

__Disclaimer : All the notebooks set root folders to "teamExperiments" at their beginning. Please make sure to change them to your teamExperiments folder, which must be put in the data folder of the snake2-beta-insa-main repo.__

_Nb : classifyTargets.ipynb, makeClassifierTrainingDataset.ipynb and createClassifier.ipynb are legacy programs that better be not used. The notebooks ending with "2" correspond to much cleaner versions._

# Functions specification

## Main scripts

All the scripts can be found on teamExperiments/

### Attack Dopel

This is the main script used as package by the other functions. There can be found definitions of main datasets _(real synthetics, targets and public)_ as well as a utility function to convert .npz files to pandas dataframes.

### Make classifier training set

This package is used to create and concatenate the "non-member" and "member" parts of the data used to train the classifiers.

The makeMemberPart function has a boolean argument "Clean" to select if all the datasets in the specified folder must be used or if the user wants to exclude them based on a criteria.
Unfortunately this feature is yet to develop. 

### Train Classifers

This package trains a classifiers array on a predefined training set and list of models. Some functions are specific to the Stacking Classifier since it has to be trained after all the other models.

Then, each of the metrics chosen to evaluate classifier is put in an array, and all the arrays are assembled in a pandas Dataframe, finally exported in a csv file.

### Classify targets

This package contains utilities to drop the unwanted columns of tasks 1 to 4, and use the predict method of classifier to produce the data to be submitted for the competition.

## Other scripts

### Visualisation

This package defines color palettes and function to plot statistic distributions of time series. Two functions are defined to plot divergence histograms. The double histogram function is used to compare divergence to synthetic dataset n°X between clean and unclean sets. 

### Compare Distributions

This package contains utilities to normalize and rescale distributions for them to be compared, and the definition of the comparison metric computation itself, which turns out to be Jensen-Shannon divergence.

