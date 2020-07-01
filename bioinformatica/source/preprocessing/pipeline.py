from bioinformatica.source.preprocessing.correlation import filter_uncorrelated, filter_correlated_features
from bioinformatica.source.preprocessing.elaboration import balance, robust_zscoring, drop_constant_features
from bioinformatica.source.preprocessing.feature_selection import boostaroota, boruta
from bioinformatica.source.preprocessing.imputation import nan_check, nan_filter, imputation
from bioinformatica.source.datasets.loader import get_data


def epigenomic_preprocessing(dataset, labels, p_value_threshold, min_correlation, correlation_threshold):
    if nan_check(dataset):
        dataset, labels = nan_filter(dataset, labels)
        dataset = imputation(dataset)

    dataset = drop_constant_features(dataset)
    dataset = robust_zscoring(dataset)

    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, min_correlation)
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)

    dataset = boruta(dataset, labels, 300, 0.05, 2)

    return dataset, labels


def pipeline(retrieve_parameters):

    load_parameters, random_state = retrieve_parameters

    p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95

    dataset, labels = get_data(load_parameters)

    if load_parameters[-1] == 'epigenomic':
        dataset, labels = epigenomic_preprocessing(dataset, labels, p_value_threshold, min_correlation,
                                                   correlation_threshold)
    else:
        dataset, labels = get_data(load_parameters)
    a = 0
    return dataset, labels
