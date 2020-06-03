import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.utils import resample, shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

random_state = 1

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
                            random_state=random_state, n_samples=max_unbalance - minority_count)], axis=0), random_state=random_state)

def data_decomposition(epigenomes, label):
    decompositions = []
    algorithms = [PCA, TSNE]
    n_components = 2
    for algorithm in algorithms:
        decomposition = algorithm(n_components=n_components, random_state=random_state)
        decompositions.append(decomposition.fit_transform(epigenomes))
    pass



