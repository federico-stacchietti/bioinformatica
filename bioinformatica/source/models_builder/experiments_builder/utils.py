from dataclasses import dataclass

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

statistical_tests = [
    wilcoxon
]

metrics = [
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
]


@dataclass
class ModelInfo:
    def __init__(self, algorithm, parameters, scores):
        self.algorithm = algorithm
        self.parameters = ' ,'.join([str(parameter) + ' : ' +
                                     str(parameters.get(parameter)) for parameter in parameters])
        for metric, score in zip(metrics, scores):
            setattr(self, metric.__name__, score)