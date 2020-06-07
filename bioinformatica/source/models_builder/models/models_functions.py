from bioinformatica.source.models_builder.models.models_libraries import *


def build_SVM(hyperparameters):
    return SVM.SVC(**hyperparameters)


def build_DecisionTree(hyperparameters):
    return DecisionTreeClassifier(**hyperparameters)


def build_RandomForest(hyperparameters):
    return RandomForestClassifier(**hyperparameters)


def build_NeuralNetwork(parameters):
    network_parameters, compiling_parameters = parameters
    model = Sequential(*network_parameters)
    model.compile(**compiling_parameters)
    return model


def train_model(is_NN, model, training_data):
    if is_NN:
        X_train, y_train, X_test, y_test, training_parameters = training_data
        return \
            model.fit(X_train, y_train, **training_parameters)
    else:
        X_train, y_train = training_data
        return model.fit(X_train, y_train)


def test_model(trained_model, X_test):
    return trained_model.predict(X_test)
