import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# restituisce True se nel dataset è presente almeno un valore Nan
def detect_nan(dataset: pd.DataFrame) -> bool:
    return sum(sample.isna().values.sum() for _, sample in dataset) > 0


# restituisce True se nel dataset è presente almeno un valore Nan
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
    return indexes


def nan_filter(dataset: pd.DataFrame, labels: np.array) -> (pd.DataFrame, np.array):
    indexes = detect_nan_in_row(dataset, int((dataset.shape[1]/10)*9))
    dataset = dataset.dropna(axis=0, thresh=int((dataset.shape[1]/10)*9))
    labels = np.delete(arr=labels, obj=indexes)
    dataset = dataset.dropna(axis=1, thresh=int((dataset.shape[0]/10)*9))
    return dataset, labels


def imputation(dataset: pd.DataFrame) -> pd.DataFrame:
    if detect_nan(dataset):
        dataset = pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(dataset.values),
            columns=dataset.columns,
            index=dataset.index
        )
    return dataset