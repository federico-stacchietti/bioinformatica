from bioinformatica.source.preprocessing.imputation import *
from bioinformatica.source.datasets.loader import *


def test_nan_check():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data

    dataset = dataset.head(100)
    nan_check(dataset)
    assert dataset is not None, 'error while executing nan check'


def test_nan_filter():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data

    dataset = dataset.head(100)
    labels = labels[:100]
    dataset, labels = nan_filter(dataset, labels)
    assert dataset is not None, 'error while filterning nans from dataset'


def test_imputation():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data

    dataset = dataset.head(100)
    assert dataset is not None, 'error while imputing dataset'

