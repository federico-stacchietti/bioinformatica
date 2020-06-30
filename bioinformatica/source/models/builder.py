from ..models.intialization_functions import *
from ..experiments.utils import metrics
from bioinformatica.source.commons import *
import numpy as np
import pandas as pd


build_models = {
    'DecisionTree': build_DecisionTree,
    'RandomForest': build_RandomForest,
    'NN': build_NeuralNetwork
}


class Model:
    def __init__(self, algorithm: str,  name: str, isNN: bool = False):
        self.__name = name
        self.__algorithm = algorithm
        self.__is_NN = isNN
        self.__training_parameters = None
        self.__model = None
        self.__trained_model = None
        self.__scores = {metric[0].__name__: [] for metric in metrics}
        if isNN:
            self.__weights = None

    def build(self, parameters: Dict or Tuple[Dict, Dict]):
        if self.__is_NN:
            self.__training_parameters = parameters[-1]
            self.__model = build_models.get(self.__algorithm)(parameters[:-1])
            self.__weights = self.__model.get_weights()
        else:
            self.__model = build_models.get(self.__algorithm)(parameters)

    def train(self, training_data: Tuple[np.array, np.array]):
        if self.__is_NN:
            self.__model.set_weights(self.__weights)
            train_model(self.__is_NN, self.__model, (*training_data, self.__training_parameters))
            self.__trained_model = self.__model
        else:
            self.__trained_model = train_model(self.__is_NN, self.__model, training_data)

    def predict(self, X: np.array or pd.Dataframe) -> np.array:
        return self.__trained_model.predict(X)

    def test_metrics(self, metric: Tuple[Callable, str], y_s: Tuple[np.array, np.array]):
        y_truth, y_prediction = y_s
        if metric[1] == 'labels':
            self.__scores.get(metric[0].__name__).append(metric[0](y_truth, np.round(y_prediction)))
        else:
            self.__scores.get(metric[0].__name__).append(metric[0](y_truth, y_prediction))

    def get_model(self) -> Any:
        return self.__model

    def get_trained_model(self) -> Any:
        return self.__trained_model

    def get_scores(self) -> Dict:
        return self.__scores

    def get_name(self) -> str:
        return self.__name
