from .utils import *
from ..datasets.loader import get_holdouts
from ..preprocessing.pipeline import pipeline
from ..models.builder import Model
from ..models.definition import define_models
from ..experiments.evaluation import evaluate


class Experiment:
    def __init__(self, algorithms, data_type, data_parameters, holdout_parameters, alphas):
        self.__data_type = data_type
        self.__data_parameters, self.__holdout_parameters = data_parameters, holdout_parameters
        self.__results = {model: [] for model in algorithms}
        self.__best_scores = {model: [] for model in algorithms}
        self.__best_model = None
        self.__alphas = alphas
        self.__statistical_tests_scores = {}

    def execute(self):
        dataset, labels = pipeline(((self.__data_parameters, self.__data_type), self.__holdout_parameters[-1]))
        for holdout in get_holdouts(dataset, labels, self.__holdout_parameters):
            training_data, test_data = holdout
            defined_algorithms = define_models(self.__data_type, len(dataset.columns))
            for algorithm in defined_algorithms:
                for hyperparameters in defined_algorithms.get(algorithm):
                    model = Model(algorithm, algorithm == 'NN', training_data, test_data)
                    model.build(hyperparameters)
                    model.train()
                    self.__results.get(algorithm).append(ModelInfo(algorithm, hyperparameters,
                                                    [model.metrics(metric) for metric in metrics], model.get_model()))

    def best_scores(self):
        for algorithm in self.__results:
            model_best_scores = []
            for metric in metrics:
                best_score, best_model = 0, None
                for model_info in self.__results.get(algorithm):
                    model_current_score = getattr(model_info, metric[0].__name__)
                    if model_current_score > best_score:
                        best_score, best_model = model_current_score, model_info
                model_best_scores.append((metric[0].__name__, best_score, best_model))
            self.__best_scores.get(algorithm).append(model_best_scores)

    def print_best_models(self):
        self.best_scores()
        for algorithm in self.__best_scores:
            for models_info in self.__best_scores.get(algorithm):
                for metric, score, model in models_info:
                    if model:
                        print_model(algorithm, metric, score, model)

    def get_results(self):
        return self.__results

    def evaluate(self):
        scores = self.__results.values()
        for alpha in self.__alphas:
            for statistical_test in statistical_tests:
                self.__statistical_tests_scores[statistical_test.__name__] = []
                for metric in metrics:
                    self.__statistical_tests_scores.get(statistical_test) \
                        .append(evaluate(scores, statistical_test, metric, alpha))
