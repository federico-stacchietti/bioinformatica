from ..models.intialization_functions import *
from ..experiments.utils import metrics
import numpy as np


build_models = {
    'SVM': build_SVM,
    'DecTree': build_DecisionTree,
    'RandForest': build_RandomForest,
    'NN': build_NeuralNetwork
}


class Model:
    def __init__(self, algorithm, isNN):
        self.__algorithm = algorithm
        self.__is_NN = isNN
        self.__training_parameters = None
        self.__model = None
        self.__trained_model = None
        self.scores = {metric[0].__name__: [] for metric in metrics}

    def build(self, parameters):
        if self.__is_NN:
            self.__training_parameters = parameters[-1]
            self.__model = build_models.get(self.__algorithm)(parameters[:-1])
        else:
            self.__model = build_models.get(self.__algorithm)(parameters)

    def train(self, training_data):
        if self.__is_NN:
            train_model(self.__is_NN, self.__model, (*training_data, self.__training_parameters))
            self.__trained_model = self.__model
        else:
            self.__trained_model = train_model(self.__is_NN, self.__model, training_data)

    def test_metrics(self, metric, test_data):
        X_test, y_test = test_data
        if metric[1] == 'labels':
            self.scores.get(metric[0].__name__)\
                .append(metric[0](y_test, np.round(test(self.__trained_model, X_test))))
        else:
            self.scores.get(metric[0].__name__).append(metric[0](y_test, test(self.__trained_model, X_test)))

    def get_model(self):
        return self.__model
