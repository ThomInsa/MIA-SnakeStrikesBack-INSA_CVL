from scipy.spatial.distance import jensenshannon
from attackDopel import *

def normalize_columns(df):
    return df.div(df.sum(axis=0), axis=1)

def rescale_distribution(scaled_df, target_df):
    """
    Adjust two data frames representing probability distributions to have the same length.

    Parameters:
        scaled_df (pd.DataFrame): The first data frame.
        target_df (pd.DataFrame): The second data frame.
        target_length (int, optional): The desired length for both datasets.
                                       Defaults to the longer data frame's length.

    Returns:
        pd.DataFrame, pd.DataFrame: Two data frames of the same length, with resampled data.
    """
    # Determine the target length
    target_length = min(len(scaled_df), len(target_df))

    # Helper function to resample a data frame
    def resample_distribution(df, length):
        weights = np.ones(len(df)) / len(df)  # Uniform weights for original distribution
        indices = np.random.choice(df.index, size=length, replace=True, p=weights)
        return df.loc[indices].reset_index(drop=True)

    # Resample data frames to the target length
    scaled_df_resampled = resample_distribution(scaled_df, target_length)

    return scaled_df_resampled

def getJensenShannonDivergences(targetDataSet, evaluationDataSet):
    targetDataSet_n = normalize_columns(targetDataSet)
    evaluationDataSet_n = normalize_columns(evaluationDataSet)
    js_distances = {}

    if len(targetDataSet_n) == len(evaluationDataSet_n):
        for col in targetDataSet_n.columns:
            js_distances[col] = jensenshannon(targetDataSet_n[col], evaluationDataSet_n[col])
    else:
        if len(targetDataSet_n) > len(evaluationDataSet_n):
            targetDataSet_n_r = rescale_distribution(targetDataSet_n, evaluationDataSet_n)
            return getJensenShannonDivergences(targetDataSet_n_r, evaluationDataSet_n)
        else:
            evaluationDataSet_n_r = rescale_distribution(evaluationDataSet_n, targetDataSet_n)
            return getJensenShannonDivergences(targetDataSet_n, evaluationDataSet_n_r)
    return js_distances

def getDatasetsFromFilePath(filePath_ToNPZ):
    datasets = {}
    for file in os.listdir(filePath_ToNPZ):
        if file.endswith(".npz"):
            index = file.replace(".npz", "")
            datasets[index] = npzPrivateToDataframe(os.path.join(filePath_ToNPZ, file))
    return datasets