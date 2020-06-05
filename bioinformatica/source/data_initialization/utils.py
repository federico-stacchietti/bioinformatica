from epigenomic_dataset import load_epigenomes

from sklearn.model_selection import StratifiedShuffleSplit


def load_dataset(cell_line, window_size, type):
    epigenomes, labels = load_epigenomes(
        cell_line=cell_line,
        dataset='fantom',
        regions=type,
        window_size=window_size
    )
    labels = labels.values.ravel()
    return epigenomes, labels


def get_holdouts(n_split, test_size, random_state):
    return StratifiedShuffleSplit(n_splits=n_split, test_size=test_size, random_state=random_state)