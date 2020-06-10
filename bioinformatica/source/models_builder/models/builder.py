from bioinformatica.source.models_builder.models.intialization_functions import *
import numpy as np


build_models = {
    'SVM': build_SVM,
    'DecTree': build_DecisionTree,
    'RandForest': build_RandomForest,
    'NN': build_NeuralNetwork
}


class Model:
    def __init__(self, type, isNN, training_set, test_set):
        self.__type = type
        self.__is_NN = isNN
        self.__training_set, self.__test_set = training_set, test_set
        self.__X_test, self.__y_test = self.__test_set
        self.__training_parameters = None
        self.__model = None
        self.__trained_model = None
        self.__scores = []

    def build(self, parameters):
        if self.__is_NN:
            self.__training_parameters = parameters[-1]
            self.__model = build_models.get(self.__type)(parameters[:-1])
        else:
            self.__model = build_models.get(self.__type)(parameters)

    def train(self):
        if self.__is_NN:
            training_data = (*self.__training_set, self.__training_parameters)
            train_model(self.__is_NN, self.__model, training_data)
            self.__trained_model = self.__model
        else:
            self.__trained_model = train_model(self.__is_NN, self.__model, self.__training_set)

    def metrics(self, metric):
        if metric[1] == 'labels':
            return metric[0](self.__y_test, np.round(test_model(self.__trained_model, self.__X_test)))
        else:
            return metric[0](self.__y_test, test_model(self.__trained_model, self.__X_test))

    def get_type(self):
        return self.__type

    type = property(get_type)

