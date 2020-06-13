from .utils import *
from ..datasets.loader import get_holdouts
from ..preprocessing.pipeline import pipeline
from ..models.builder import Model
from ..models.definition import define_models
from ..experiments.evaluation import test_models


class Experiment:
    def __init__(self, algorithms, data_type, data_parameters, holdout_parameters, alphas):
        self.__data_type = data_type
        self.__data_parameters, self.__holdout_parameters = data_parameters, holdout_parameters
        self.__models = []
        self.__results = {model: [] for model in algorithms}
        self.__best_scores = {model: [] for model in algorithms}
        self.__best_model = None
        self.__alphas = alphas
        self.__statistical_tests_scores = {}

    def execute(self):
        dataset, labels = pipeline(((self.__data_parameters, self.__data_type), self.__holdout_parameters[-1]))
        defined_algorithms = define_models(self.__data_type, len(dataset.columns))
        for algorithm in defined_algorithms:
            for hyperparameters in defined_algorithms.get(algorithm):
                model = Model(algorithm, algorithm == 'NN')
                model.build(hyperparameters)
                self.__models.append(model)
        for holdout, data in enumerate(get_holdouts(dataset, labels, self.__holdout_parameters)):
            for model in self.__models:
                training_data, test_data = data
                model.train(training_data)
                for metric in metrics:
                    model.test_metrics(metric, test_data)

    def evaluate(self):
        for alpha in self.__alphas:
            for statistical_test in statistical_tests:
                self.__statistical_tests_scores[statistical_test.__name__] = []
                for metric in metrics:
                    self.__statistical_tests_scores.get(statistical_test.__name__) \
                        .append(test_models(self.__models, statistical_test, metric, alpha))


