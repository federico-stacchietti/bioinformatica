from bioinformatica.source.experiments.models.models_libraries import *
from bioinformatica.source.experiments.models.models_functions import *

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score


build_models = {
    'SVM': build_SVM,
    'DecTree': build_DecisionTree,
    'RandForest': build_RandomForest,
    'NN': build_NeuralNetwork
}

metrics = [
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
]


class Model:
    def __init__(self, type, training_set, test_set, training_parameters=None):
        self._type = type
        if self._type in ['SVM', 'DecTree', 'RandForest']:
            self._is_NN = False
        else:
            self._is_NN = True
        self._training_set, self._test_set = training_set, test_set
        self._training_parameters = training_parameters
        self._model = None
        self._trained_model = None
        self._scores = []

    def build(self, parameters):
        self._model = build_models.get(self._type)(parameters)

    def train(self):
        if self._is_NN:
            training_data = (*self._training_set, *self._test_set, self._training_parameters)
            self._trained_model = train_model(self._is_NN, self._model, training_data)
        else:
            self._trained_model = train_model(self._is_NN, self._model, self._training_set)

    def model_accuracy(self):
        for metric in metrics:
            self._scores.append(metric(self._test_set[-1], test_model(self._trained_model)))



