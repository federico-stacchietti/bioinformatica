import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.utils import resample, shuffle


def detect_NaN(epigenomes):
    return sum(sample.isna().values.sum() for _, sample in epigenomes)

def imputation(epigenomes):
    if detect_NaN(epigenomes) > 0:
        epigenomes = pd.DataFrame(
            KNNImputer(n_neighbors=20).fit_transform(epigenomes.values),
            columns=epigenomes.columns,
            index=epigenomes.index
        )
    return epigenomes

def check_balance(epigenomes, labels):
    n_samples = len(epigenomes)
    max_unbalance = n_samples // 100
    counts = [(labels.count(label), label) for label in labels]
    minority_count, minority_label = min(counts, key=lambda x: x[0])
    if minority_count < max_unbalance:
        majority_class = epigenomes.iloc[np.where(labels != minority_label)]
        # trova i sample della classe sbilanciata, gli fa il resample e lo unisce alla classe non sbilanciata, poi fa lo shuffle
        epigenomes = shuffle(pd.concat([majority_class, resample(epigenomes.iloc[np.where(labels == minority_label)],
                            random_state=0, n_samples=max_unbalance - minority_count)], axis=0), random_state=0)

# Input:dataframe. Controlla la presenza di feature costanti. Se presenti, le elimina dal dataframe e lo restituisce
def drop_constant_features(epigenomes):
    const_features = [feature for feature in epigenomes.columns if epigenomes[feature].nunique() == 1]
    non_const_features = [feature for feature in epigenomes.columns if feature not in const_features]
    if not non_const_features:
        pass
        # print('there is no constant feature in the dataset')
    else:
        return(epigenomes[non_const_features])




