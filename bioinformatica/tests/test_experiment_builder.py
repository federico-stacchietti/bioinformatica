from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.utils import metrics, statistical_tests
from bioinformatica.tests.dummy_models import define_models


def test_experiment():
    data_type = 'epigenomic'
    cell_line, window_size, typez = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 2, 0.2, 1
    balance = None
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, typez), data_type)
    alphas = [0.05]
    defined_algorithms = define_models()
    experiment = Experiment(data_type, data_parameters, holdout_parameters, alphas, defined_algorithms, balance)

    experiment.execute()
    assert len(experiment.get_models()) == len(defined_algorithms.values()), 'not all models were builder'
    total_scores = 0
    for model in experiment.get_models():
        total_scores += len([value for scores in model.get_scores().values() for value in scores])
    assert total_scores == n_split * len(metrics) * 2 * len(experiment.get_models()), \
        'not all models were correctly tested'

    experiment.evaluate()
    assert len(experiment.get_best_models().items()) * len(metrics) == len(statistical_tests) * len(metrics) * len(alphas), \
        'not all statistical tests were executed correctly or some alpha value may not have been tested'

    try:
        experiment.get_models()
    except:
        'error in models retrieving'

    try:
        experiment.get_best_models()
    except:
        'error in models retrieving'

    try:
        experiment.print_model_info()
    except:
        'error in printing models'