import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.utils import resample, shuffle


def detect_NaN(dataset):
    return sum(sample.isna().values.sum() for _, sample in dataset)


def imputation(dataset):
    if detect_NaN(dataset) > 0:
        dataset = pd.DataFrame(
            KNNImputer(n_neighbors=20).fit_transform(dataset.values),
            columns=dataset.columns,
            index=dataset.index
        )
    return dataset


def balance(dataset, labels, random_state):
    n_samples = len(dataset)
    max_unbalance = n_samples // 100
    counts = [(labels.count(label), label) for label in labels]
    minority_count, minority_label = min(counts, key=lambda x: x[0])
    if minority_count < max_unbalance:
        majority_class = dataset.iloc[np.where(labels != minority_label)]
        to_sample = max_unbalance - minority_count
        # trova i sample della classe sbilanciata, gli fa il resample e lo unisce alla classe non sbilanciata,
        # poi fa lo shuffle
        dataset = shuffle(pd.concat([majority_class, resample(dataset.iloc[np.where(labels == minority_label)],
                          random_state=random_state, n_samples=to_sample)], axis=0), random_state=random_state)
    return dataset