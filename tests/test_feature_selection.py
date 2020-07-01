from bioinformatica.source.preprocessing.feature_selection import *
from bioinformatica.source.datasets.loader import *
from bioinformatica.source.preprocessing.imputation import imputation


def test_boruta(dataset: pd.DataFrame, labels: np.array):
    dataset = boruta(dataset, labels, 50, 0.01, 42)
    assert isinstance(dataset, pd.DataFrame), 'boruta returned type is not pd.DataFrame'


def test_execution():
    parameters = ('GM12878', 200, 'enhancers'), 'epigenomic'
    dataset, labels = get_data(parameters)

    dataset = imputation(dataset)

    test_boruta(dataset, labels)

