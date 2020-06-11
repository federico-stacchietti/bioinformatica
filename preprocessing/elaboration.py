import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample


def drop_constant_features(dataset: pd.DataFrame) -> pd.DataFrame:
    non_const_features = [feature for feature in dataset.columns if dataset[feature].nunique() > 1]
    return dataset[non_const_features]


def rebalance_classes(dataset: pd.DataFrame, labels: np.array, random_state: int) -> (pd.DataFrame, np.array):
    n_samples = len(dataset)
    max_unbalance = n_samples // 10
    unique, counts = np.unique(labels, return_counts=True)
    minority_label, minority_count = min(zip(unique, counts), key=lambda x: x[1])
    if minority_count < max_unbalance:
        dataset = pd.concat([dataset, resample(dataset.iloc[np.where(labels == minority_label)],
                                    random_state=random_state, n_samples=max_unbalance - minority_count)], axis=0)
        labels = labels + np.full(max_unbalance - minority_count, minority_label)
        np.random.seed(random_state)
        shuffle = np.random.permutation(range(len(labels)))
        dataset, labels = dataset[shuffle], labels[shuffle]
    return dataset, labels


def robust_zscoring(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(dataset.values),
        columns=dataset.columns,
        index=dataset.index
    )