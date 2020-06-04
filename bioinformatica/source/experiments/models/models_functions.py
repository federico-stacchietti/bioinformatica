from bioinformatica.source.experiments.models.models_libraries import *


def build_SVM(hyperparameters):
    return SVM(hyperparameters)


def build_DecisionTree(hyperparameters):
    return DecisionTreeClassifier(hyperparameters)


def build_RandomForest(hyperparameters):
    return RandomForestClassifier(hyperparameters)


def build_NeuralNetwork(hyperparameters):
    network_parameters, compiling_parameters = hyperparameters
    model = Sequential(network_parameters)
    return model.compile(compiling_parameters)


def train_model(is_NN, model, training_data):
    if is_NN:
        X_train, y_train, X_test, y_test, training_parameters = training_data
        return \
            model.fit(X_train, y_train, training_parameters, validation_data=(X_test, y_test))
    else:
        X_train, y_train = training_data
        return model.fit(X_train, y_train)


def test_model(trained_model, X_test):
    return trained_model.predict(X_test)


