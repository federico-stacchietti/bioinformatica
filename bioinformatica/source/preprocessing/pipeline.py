from ..preprocessing.correlation import filter_uncorrelated, filter_correlated_features
from ..preprocessing.elaboration import balance, robust_zscoring, drop_constant_features
from ..preprocessing.feature_selection import boostaroota
from ..preprocessing.imputation import nan_check, nan_filter, imputation
from ..datasets.loader import get_data
from bioinformatica.source.utils import *
import pandas as pd
import numpy as np


def epigenomic_preprocessing(dataset: pd.DataFrame, labels: np.array, random_state: int, p_value_threshold: float, min_correlation: float, correlation_threshold: float) -> Tuple[pd.DataFrame, np.array]:
    if nan_check(dataset):
        dataset, labels = nan_filter(dataset, labels)
        dataset = imputation(dataset)

    dataset = drop_constant_features(dataset)
    dataset = robust_zscoring(dataset)

    dataset = filter_uncorrelated(dataset, labels, p_value_threshold, min_correlation)
    dataset = filter_correlated_features(dataset, p_value_threshold, correlation_threshold)

    dataset = boostaroota(dataset, labels)
    return dataset, labels


def sequences_preprocessing(dataset: pd.DataFrame, labels: np.array, random_state: int) -> Tuple[pd.DataFrame, np.array]:
    return dataset, labels


def pipeline(data_parameters: Tuple[Tuple[Tuple[str, int, str], str], int]) -> Tuple[pd.DataFrame, np.array]:
    # load_parameters, random_state = data_parameters
    # dataset, labels = get_data(load_parameters)
    #
    # if load_parameters[-1] == 'epigenomic':
    #     p_value_threshold, min_correlation, correlation_threshold = 0.01, 0.05, 0.95
    #     return epigenomic_preprocessing(dataset, labels, random_state, p_value_threshold, min_correlation,
    #                                     correlation_threshold)
    # else:
    #     return sequences_preprocessing(dataset, labels, random_state)

    dataset = pd.read_csv("/home/flavio/boruta/dataset_borutaK562.csv")
    f = open('/home/flavio/boruta/labels_borutaK562.txt', 'r+')
    labels = np.asarray([int(line) for line in f.readlines()])
    f.close()
    dataset.drop(dataset.columns[0], inplace=True, axis=1)
    # dataset = pd.DataFrame(PCA_function(dataset, 20))
    # dataset, labels = balance(dataset, labels, 42)
    return dataset, labels