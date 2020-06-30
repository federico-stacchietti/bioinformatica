from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def PCA_function(dataset, n_components):
    return PCA(n_components=n_components).fit_transform(dataset)


def TSNE_function(dataset, n_components, perplexity, learning_rate):
    return TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)\
        .fit_transform(dataset)
