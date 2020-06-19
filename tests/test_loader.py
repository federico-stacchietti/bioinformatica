from math import ceil
from bioinformatica.source.datasets.loader import *
from bioinformatica.source.utils import *

def test_get_data(parameters: Tuple[Tuple[str, int, str], str]):
    data = get_data(parameters)
    dataset, labels = data
    assert len(data) == 2, 'error in loading data'
    assert isinstance(dataset, pd.DataFrame), 'error in loading dataset'
    assert isinstance(labels, np.ndarray), 'error in loading labels'


def test_get_holdouts(dataset: pd.DataFrame, labels: np.array, holdout_parameters: Tuple[int, float, int]):
    n_split, test_size, random_state = holdout_parameters
    for training, test in get_holdouts(dataset, labels, holdout_parameters):
        training_set, training_labels = training
        test_set, test_labels = test
        assert len(test_set) == ceil(len(dataset) * test_size), 'error in creating training and test set'
        assert len(dataset) == len(training_set) + len(test_set), 'error in creating training and test set'


if __name__ == "__main__":
    parameters = ('K562', 200, 'enhancers'), 'epigenomic'
    data = get_data(parameters)
    dataset, labels = data
    holdout_parameters = 5, 0.2, 42
    # test_get_data(parameters)
    # test_get_holdouts(dataset, labels, holdout_parameters)
    training, test = next(get_holdouts(dataset, labels, holdout_parameters))
    training_set, training_labels = training
    test_set, test_labels = test
    print(len(test_set), len(dataset) * holdout_parameters[1])

