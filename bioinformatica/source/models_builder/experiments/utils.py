from dataclasses import dataclass

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from pprint import pprint


statistical_tests = [
    wilcoxon
]

metrics = [
    (accuracy_score, 'labels'),
    (balanced_accuracy_score, 'labels'),
    (roc_auc_score, 'prob'),
    (average_precision_score, 'labels')
]


@dataclass
class ModelInfo:
    def __init__(self, algorithm, parameters, scores, model=None):
        self.algorithm = algorithm
        self.parameters = parameters
        for metric, score in zip(metrics, scores):
            setattr(self, metric[0].__name__, score)
        if self.algorithm == 'NN':
            self.nn_model = model
        else:
            self.nn_model = None


def print_model(algorithm, metric, score, model):
    if not algorithm == 'NN':
        print(algorithm, '\n', metric, score)
        pprint(model.parameters)
        print('\n---------------------------\n')
    else:
        print(metric, score, '\n', model.nn_model.summary())
        pprint(model.parameters[1])
        pprint(model.parameters[-1])
        for obj in model.parameters[-1].get('callbacks'):
            pprint(vars(obj))
        print('\n---------------------------\n')
