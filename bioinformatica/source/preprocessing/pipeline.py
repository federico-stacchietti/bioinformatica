from ..preprocessing.correlation import filter_uncorrelated, filter_correlated_features
from ..preprocessing.elaboration import balance, robust_zscoring, drop_constant_features
from ..preprocessing.feature_selection import boostaroota
from ..preprocessing.imputation import nan_check, nan_filter, imputation
from ..datasets.loader import get_data


def epigenomic_preprocessing(dataset, labels, random_state, p_value_threshold, min_correlation, correlation_threshold):

    if nan_check(dataset):
        dataset, labels = nan_filter(dataset, labels)
        dataset = imputation(dataset)

    dataset, labels = balance(dataset, labels, random_state)
    dataset = drop_constant_features(dataset)
    dataset = robust_zscoring(dataset)

    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, min_correlation)
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)

    dataset = boostaroota(dataset, labels)
    return dataset, labels


def sequences_preprocessing(dataset, labels, random_state):
    dataset, labels = balance(dataset, labels, random_state)
    return dataset, labels


def pipeline(data_parameters):
    load_parameters, data_type, random_state = data_parameters
    dataset, labels = get_data(data_parameters)

    if data_type == 'epigenomic':
        p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95
        return epigenomic_preprocessing(dataset, labels, random_state, p_value_threshold, min_correlation, correlation_threshold)
    else:
        return sequences_preprocessing(dataset, labels, random_state)