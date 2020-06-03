from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def data_decomposition(epigenomes, algorithms):
    decompositions = []
    n_components = 2
    if 'PCA' in algorithms:
        decompositions.append(('PCA', exec_PCA(epigenomes, n_components)))
    if 'TSNE' in algorithms:
        threshold = 50
        perplexities, learning_rate = [10, 20, 50, 100, 500, 1000], 200
        if threshold > epigenomes.shape[1]:
            for perplexity in perplexities:
                decompositions.append(('TSNE', exec_TSNE(exec_PCA(epigenomes, threshold),
                                                         n_components, perplexity, learning_rate)))
        else:
            for perplexity in perplexities:
                decompositions.append(('TSNE', exec_TSNE(epigenomes, n_components, perplexity, learning_rate)))
    return decompositions
    pass


def exec_PCA(dataset, n_components):
    return PCA(n_components=n_components).fit_transform(dataset)


def exec_TSNE(dataset, n_components, perplexity, learning_rate):
    return TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)\
        .fit_transform(dataset)