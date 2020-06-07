from bioinformatica.source.data_initialization.datasets_initialization import get_data
from bioinformatica.source.models_builder.models.init_models import Model
from bioinformatica.source.models_builder.models.models_definition import define_models
from bioinformatica.source.models_builder.experiments_builder.experiments_utils import *


class Experiment:
    def __init__(self, models, data_type, experiment_params):
        self.__data_type = data_type
        self.__experiment_params = experiment_params
        self.__holdouts = get_data(data_type)(experiment_params)
        self.__models = models
        self.__results = {model: [] for model in self.__models}

        self.__best_scores = {model: [] for model in self.__models}
        self.__worst_scores = {}

        self.__best_model = None

    def execute(self):
        for holdout in self.__holdouts:
            training_data, test_data = holdout
            algorithms = define_models()
            for algorithm in algorithms:
                for hyperparameters_list, training_parameters in algorithms.get(algorithm):
                    model = Model(algorithm, algorithm == 'NN', training_data, test_data, training_parameters)
                    for hyperparameters in hyperparameters_list:
                        model.build(hyperparameters)
                        model.train()
                        self.__results.get(algorithm).append(ModelInfo(hyperparameters,
                                                                       [model.metrics(metric) for metric in metrics]))

    def best_by_algorithm(self):
        for algorithm in self.__results:
            for metric in metrics:
                model_best_scores = []
                best_score, best_parameters = 0, None
                for model_info in self.__results.get(algorithm):
                    if model_info.metric.__name__ > best_score:
                        best_score, best_parameters = model_info.metric.__name__, model_info.parameters
                model_best_scores.append((metric.__name__, best_score, best_parameters))
                self.__best_scores.get(algorithm).append(model_best_scores)
        return self.__best_scores

    def get_results(self):
        return self.__results

    best_scores = property(best_by_algorithm)
    all_results = property(get_results)
