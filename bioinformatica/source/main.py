from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.models.definition import *


if __name__ == '__main__':
    # pass
    experiment_id = 2
    data_type = 'sequences'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')
    # experiment.results_to_dataframe()
