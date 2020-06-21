from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.preprocessing.pipeline import *
import time
from pprint import pprint
from scipy.stats import wilcoxon
import random
from skopt import gp_minimize


if __name__ == '__main__':
    # algorithms = ['AdaBoost', 'GradBoost']

    algorithms = ['DecTree', 'Rand']
    data_type = 'epigenomic'
    cell_line, window_size, typez = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 2, 0.2, 42
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = (cell_line, window_size, typez)
    alphas = [0.05]
    experiment = Experiment(algorithms, data_type, data_parameters, holdout_parameters, alphas)
    experiment.execute()
    # experiment.evaluate()

    print('\n\n++++++++SCORES++++++++\n\n')
    for x in experiment.modelz:
        pprint(x.get_model())
        pprint(x.scores)
        print('\n-------------------------------\n')

    # print('\n\n++++++++BEST MODELS++++++++\n')
    # for x in experiment.scors:
    #     for alpha, metric, mean_score, model in experiment.scors.get(x):
    #         print(alpha, metric, mean_score)
    #         pprint(model.get_model())
    #         print(model.scores.get(metric))
    #         print('\n\n')

    res = gp_minimize(func=,
                      dimensions=,
                      base_estimator=None,
                      n_calls=100,
                      n_random_starts=10,
                      acq_func='gp_edge',
                      acq_optimizer='auto',
                      x0=None,
                      y0=None,
                      random_state=random_state,
                      callback=None,
                      n_points=10000,
                      n_restarts_optimizer=5,
                      xi=0.01,
                      kappa=1.96,
                      noise='gaussians',
                      n_jobs=-1,
                      model_queue_size=None)
