from bioinformatica.source.preprocessing.correlation import *
from bioinformatica.source.datasets.loader import *
from bioinformatica.source.preprocessing.imputation import imputation


def test_uncorrelated_test(dataset: pd.DataFrame, labels: np.array, p_value_threshold: float):
    dataset = uncorrelated_test(dataset, labels, p_value_threshold)
    assert isinstance(dataset, list), 'uncorrelated_test returned type is not list'


def test_mic(dataset: pd.DataFrame, labels: np.array):
    dataset = mic(dataset, labels)
    assert isinstance(dataset, dict), 'mic returned type is not dict'


def test_filter_uncorrelated(dataset: pd.DataFrame, labels: np.array, p_value_threshold: float, correlation_threshold: float):
    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, correlation_threshold)
    assert isinstance(dataset, pd.DataFrame), 'filter_uncorrelated returned type is not pd.DataFrame'


def test_filter_correlated_features(dataset: pd.DataFrame, p_value_threshold: float, correlation_threshold: float):
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)
    assert isinstance(dataset, pd.DataFrame), 'filter_correlated_features returned type is not pd.DataFrame'


def test_feature_correlation(dataset: pd.DataFrame):
    dataset = feature_correlation(dataset)
    assert isinstance(dataset, list), 'feature_correlation returned type is not list'


def test_execution():
    parameters = ('GM12878', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data
    dataset = imputation(dataset)
    p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95
    test_uncorrelated_test(dataset, labels, p_value_threshold)
    test_mic(dataset, labels)
    test_filter_uncorrelated(dataset, labels, p_value_threshold, correlation_threshold)
    test_filter_correlated_features(dataset, p_value_threshold, correlation_threshold)
    test_feature_correlation(dataset)
