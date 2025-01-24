import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn

# Constants for the whole competition
NUMBER_OF_COLUMNS_DATASET = 7

# Making dataframes from files so that we load them only one time

publicData_Tasks12 = pd.read_parquet('../publicData/publicDatasetTask1-2.parquet')
publicData_Tasks34 = pd.read_parquet('../publicData/publicDatasetTask3-4.parquet')
targets_Task1 = pd.read_csv('../publicData/targetsTask1.csv')
targets_Task2 = pd.read_csv('../publicData/targetsTask2.csv')
targets_Task3 = pd.read_csv('../publicData/targetsTask3.csv')
targets_Task4 = pd.read_csv('../publicData/targetsTask4.csv')


def npzPrivateToDataframe(npz_FilePath):
    npz = np.load(npz_FilePath)
    features_array = npz["data_feature"]
    # Reshape the array to remove the extra dimension
    if features_array.ndim == 3 and features_array.shape[2] == 1:
        features_array = features_array.reshape(features_array.shape[0], features_array.shape[1])
    dataFrame = pd.DataFrame(features_array)
    return dataFrame

syntheticData_Task1 = npzPrivateToDataframe("../publicData/syntheticTask1.npz")
syntheticData_Task2 = npzPrivateToDataframe('../publicData/syntheticTask2.npz')
syntheticData_Task3 = npzPrivateToDataframe('../publicData/syntheticTask3.npz')
syntheticData_Task4 = npzPrivateToDataframe('../publicData/syntheticTask4.npz')