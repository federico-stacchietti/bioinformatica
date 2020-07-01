from bioinformatica.source.preprocessing.imputation import *
from bioinformatica.source.datasets.loader import *


def test_nan_check(dataset: pd.DataFrame):
    assert isinstance(nan_check(dataset), bool), 'nan_check returned type is not bool'


def test_detect_nan_in_row(dataset: pd.DataFrame, threshold: int):
    assert isinstance(detect_nan_in_row(dataset, threshold), np.ndarray),\
        'detected_nan_in_row returned type is not np.ndarray'


def test_nan_filter(dataset: pd.DataFrame, labels: np.array):
    dataset, labels = nan_filter(dataset, labels)
    assert isinstance(dataset, pd.DataFrame), 'the first element returned by nan_filter is not pd.DataFrame'
    assert isinstance(labels, np.ndarray), 'the second element returned by nan_filter is not np.array'


def test_imputation(dataset: pd.DataFrame):
    assert isinstance(imputation(dataset), pd.DataFrame), 'imputation returned type is not pd.DataFrame'


def test_execution():
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data

    test_nan_check(dataset)
    test_detect_nan_in_row(dataset, int((dataset.shape[1] / 10) * 9))
    test_nan_filter(dataset, labels)
    test_imputation(dataset)
