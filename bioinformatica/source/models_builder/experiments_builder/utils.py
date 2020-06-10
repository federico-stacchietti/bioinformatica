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
    def __init__(self, algorithm, parameters, scores):
        self.algorithm = algorithm
        self.parameters = parameters
        for metric, score in zip(metrics, scores):
            setattr(self, metric[0].__name__, score)


def print_models(isNN, model):
    if not isNN:
        print(model.algorithm)
        pprint(model.parameters)
    else:
        print(model.summary())
        pprint(model[1])
        pprint(model[-1])
        for obj in model[-1].get('callbacks'):
            pprint(vars(obj))
