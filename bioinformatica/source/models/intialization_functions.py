from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from ..models.libraries import *


def build_SVM(hyperparameters):
    return SVM.SVC(**hyperparameters)


def build_DecisionTree(hyperparameters):
    return DecisionTreeClassifier(**hyperparameters)


def build_RandomForest(hyperparameters):
    return RandomForestClassifier(**hyperparameters)


def build_AdaBoost(hyperparameters):
    return AdaBoostClassifier(**hyperparameters)


def build_GradientBoosting(hyperparameters):
    return GradientBoostingClassifier(**hyperparameters)


def build_SGD(hyperparameters):
    return SGDClassifier(**hyperparameters)


def build_NeuralNetwork(parameters):
    network_parameters, compiling_parameters = parameters
    model = Sequential(*network_parameters)
    model.compile(**compiling_parameters)
    return model


def train_model(is_NN, model, training_data):
    if is_NN:
        X_train, y_train, training_parameters = training_data
        model.fit(X_train, y_train, **training_parameters)
    else:
        X_train, y_train = training_data
        return model.fit(X_train, y_train)


def test(trained_model, X_test):
    return trained_model.predict(X_test)