from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def PCA_function(dataset):
    return PCA(n_components=50, random_state=0).fit_transform(dataset)


def TSNE_function(dataset):
    dataset = PCA_function(dataset)
    return TSNE(n_components=2, perplexity=30, random_state=0)\
        .fit_transform(dataset)
