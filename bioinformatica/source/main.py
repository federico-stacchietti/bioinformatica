from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.models.definition import *


if __name__ == '__main__':
    data_type = 'sequences'
    cell_line, window_size, typez = 'HEK293', 200, 'enhancers'
    n_split, test_size, random_state = 4, 0.2, 1
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, typez), data_type)
    defined_algorithms = define_models()
    alphas = [0.05]
    experiment = Experiment(data_type, data_parameters, holdout_parameters, alphas, defined_algorithms)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info()
