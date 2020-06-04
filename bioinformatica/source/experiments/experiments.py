from bioinformatica.source.data_initialization.datasets_initialization import get_data
from bioinformatica.source.experiments.models.init_models import Model
from bioinformatica.source.experiments.models_definition import define_models


class Experiment:
    def __init__(self, models, data_type, experiment_params):
        self.__data_type = data_type
        self.__experiment_params = experiment_params
        self.__holdouts = get_data(data_type)(experiment_params)
        self.__models = models
        self.__results = []

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
                        self.__results.append((model.type, model.accuracy, ' '
                                               .join(parameter for parameter in hyperparameters)))


