from bioinformatica.source.preprocessing.decompositions import *
from bioinformatica.source.datasets.loader import get_data


def test_pca_function():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = PCA_function(dataset)
    assert isinstance(dataset, list), 'data_decomposition returned type is not list'


def test_tsne_function():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)
    dataset = TSNE_function(dataset)
    assert isinstance(dataset, list), 'data_decomposition returned type is not list'



