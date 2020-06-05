from dataclasses import dataclass

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

statistical_tests = {
    wilcoxon: [.01, .05, .1]
}

metrics = [
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
]


@dataclass
class ModelInfo:
    def __init__(self, parameters, scores):
        self.parameters = ' '.join(parameter for parameter in parameters)
        for metric, score in zip(metrics, scores):
            setattr(self, metric.__name__, score)