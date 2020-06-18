import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def drop_constant_features(dataset: pd.DataFrame) -> pd.DataFrame:
    non_const_features = [feature for feature in dataset.columns if dataset[feature].nunique() > 1]
    return dataset[non_const_features]


def balance(dataset: pd.DataFrame, labels: np.array, random_state: int) -> (pd.DataFrame, np.array):
    sampler = RandomOverSampler(random_state=0)
    dataset, labels = sampler.fit_resample(dataset, labels)
    return dataset, labels


def robust_zscoring(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(dataset.values),
        columns=dataset.columns,
        index=dataset.index
    )