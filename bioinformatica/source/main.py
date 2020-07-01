from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.definition import *


'''
Example of use:
    experiment_id = 1
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance, 
                            save_results)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')

'''


if __name__ == '__main__':
    experiment_id = 1
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance,
                            save_results)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')
