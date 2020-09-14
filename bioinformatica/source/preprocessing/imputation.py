import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def nan_check(dataset: pd.DataFrame) -> bool:
    for feature, x in dataset.isna().items():
        for element in x:
            if element:
                return True
    return False


def detect_nan_in_row(dataset: pd.DataFrame, threshold: int) -> np.array:
    indexes = []
    for index, row in dataset.iterrows():
        if row.count() < threshold:
            indexes.append(index)
    return np.array(indexes)


def nan_filter(dataset: pd.DataFrame, labels: np.array) -> (pd.DataFrame, np.array):
    indexes = detect_nan_in_row(dataset, int((dataset.shape[1]/10)*9))
    if len(indexes) > 0:
        dataset = dataset.dropna(axis=0, thresh=int((dataset.shape[1]/10)*9))
        labels = np.delete(arr=labels, obj=indexes)
    dataset = dataset.dropna(axis=1, thresh=int((dataset.shape[0]/10)*9))
    return dataset, labels


def imputation(dataset: pd.DataFrame) -> pd.DataFrame:
    if nan_check(dataset):
        dataset = pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(dataset.values),
            columns=dataset.columns,
            index=dataset.index
        )
    return dataset
