from bioinformatica.source.data_initialization.datasets_initialization import get_data
from bioinformatica.source.models_builder.experiments_builder.utils import *
from bioinformatica.source.models_builder.models.builder import Model
from bioinformatica.source.models_builder.models.definition import define_models
from bioinformatica.source.models_builder.experiments_builder.evaluation import evaluate


class Experiment:
    def __init__(self, algorithms, data_type, experiment_params, alphas):
        self.__data_type = data_type
        self.__experiment_params = experiment_params
        self.__holdouts = get_data(data_type)(experiment_params)
        self.__results, self.__best_scores = [{model: [] for model in algorithms}] * 2
        self.__best_model = None
        self.__alphas = alphas
        self.__statistical_tests_scores = {}

    def execute(self):
        for holdout in self.__holdouts:
            training_data, test_data = holdout
            defined_algorithms = define_models()
            for algorithm in defined_algorithms:
                for hyperparameters in defined_algorithms.get(algorithm):
                    model = Model(algorithm, algorithm == 'NN', training_data, test_data)
                    model.build(hyperparameters)
                    model.train()
                    self.__results.get(algorithm).append(ModelInfo(algorithm, hyperparameters,
                                                                   [model.metrics(metric) for metric in metrics]))

    def best_by_algorithm(self):
        for algorithm in self.__results:
            model_best_scores = []
            for metric in metrics:
                best_score, best_parameters = 0, None
                for model_info in self.__results.get(algorithm):
                    model_current_score = getattr(model_info, metric[0].__name__)
                    if model_current_score > best_score:
                        best_score, best_parameters = model_current_score, model_info.parameters
                model_best_scores.append((metric[0].__name__, best_score, best_parameters))
            self.__best_scores.get(algorithm).append(model_best_scores)
        return self.__best_scores

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

    best_scores = property(best_by_algorithm)
    all_results = property(get_results)
