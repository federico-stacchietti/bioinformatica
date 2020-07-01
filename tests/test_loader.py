from math import ceil
from bioinformatica.source.datasets.loader import *
import pytest


@pytest.fixture()
def test_get_data(parameters: Tuple[Tuple[str, int, str], str]):
    data = get_data(parameters)
    dataset, labels = data
    assert len(data) == 2, 'error while loading data'
    assert isinstance(dataset, pd.DataFrame), 'error while loading dataset'
    assert isinstance(labels, np.ndarray), 'error while loading labels'


@pytest.fixture()
def test_get_holdouts(dataset: pd.DataFrame, labels: np.array, holdout_parameters: Tuple[int, float, int], data_type: str):
    n_split, test_size, random_state = holdout_parameters
    for training, test in get_holdouts(dataset, labels, holdout_parameters, data_type):
        training_set, training_labels = training
        test_set, test_labels = test
        assert len(test_set) == ceil(len(dataset) * test_size), 'error while creating training and test set'
        assert len(dataset) == len(training_set) + len(test_set), 'error while creating training and test set'


@pytest.fixture()
def test_execution():
    parameters = (('K562', 200, 'enhancers'), 'epigenomic')
    data = get_data(parameters)
    dataset, labels = data
    holdout_parameters = 1, 0.2, 1
    test_get_data(parameters)
    test_get_holdouts(dataset, labels, holdout_parameters, parameters[-1])

