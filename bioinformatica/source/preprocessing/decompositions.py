from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


def PCA_function(dataset: pd.DataFrame or np.array) -> pd.DataFrame or np.array:
    return PCA(n_components=50, random_state=0).fit_transform(dataset)


def TSNE_function(dataset: pd.DataFrame or np.array) -> pd.DataFrame or np.array:
    dataset = PCA_function(dataset)
    return TSNE(n_components=2, perplexity=30, random_state=0)\
        .fit_transform(dataset)
