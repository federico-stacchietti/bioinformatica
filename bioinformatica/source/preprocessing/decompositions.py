from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def data_decomposition(epigenomes, algorithms):
    decompositions = []
    n_components = 2
    if 'PCA' in algorithms:
        decompositions.append(('PCA', PCA_function(epigenomes, n_components)))
    if 'TSNE' in algorithms:
        threshold = 50
        perplexities, learning_rate = [10, 20, 50, 100, 500, 1000], 200
        if threshold > epigenomes.shape[1]:
            for perplexity in perplexities:
                decompositions.append(('TSNE', TSNE_function(PCA_function(epigenomes, threshold),
                                                             n_components, perplexity, learning_rate)))
        else:
            for perplexity in perplexities:
                decompositions.append(('TSNE', TSNE_function(epigenomes, n_components, perplexity, learning_rate)))
    return decompositions


def PCA_function(dataset, n_components):
    return PCA(n_components=n_components).fit_transform(dataset)


def TSNE_function(dataset, n_components, perplexity, learning_rate):
    return TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)\
        .fit_transform(dataset)