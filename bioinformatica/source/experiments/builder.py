from bioinformatica.source.experiments.utils import *
from bioinformatica.source.datasets.loader import get_holdouts
from bioinformatica.source.preprocessing.pipeline import pipeline
from bioinformatica.source.models.builder import Model
from bioinformatica.source.experiments.evaluation import test_models
from bioinformatica.source.preprocessing.elaboration import balance
from bioinformatica.source.commons import *


'''
Experiment class
Class used to perform experiments
An experiment is intended as a cell line to be analyzed, performing data retrieval, data preprocessing and machine learning
algorithms on it.
An Experiment object takes several arguments:
- data_parameters: a tuple of tuple, to indicate cell line, window size and if the data must be enhancers or promoters and 
a string inside the external tuple to indicate if it is epigenomic or sequences dataset
- holdout_parameters: a tuple to indicate the number of split for holdouts, test set size and random state
- alphas: values to use with statistical tests, like wilcoxon
- defined_algorithms: algorithms and models defined for the experiment
- balance_type: default is None, can be set to 

methods:
- execute(): runs the experiment. Retrieves the data, create holdouts (for train and test), builds the models, execute trainings
    and test the trained models
    
- evaluate(): executes statistical tests, ordered by metric and by statistical test 

Example of use:
    data_type = 'sequences'
    cell_line, window_size, typez = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = None
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, typez), data_type)
    alphas = [0.05]
    experiment = Experiment(data_parameters, holdout_parameters, alphas, defined_algorithms, balance)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')

'''


class Experiment:
    def __init__(self, data_parameters: Tuple[Tuple[str, int, str], str], holdout_parameters: Tuple[int, float, int],
                 alphas: List[float], defined_algorithms: Dict[str, List], balance_type: str = None):
        self.__data_type = data_parameters[1]
        self.__data_parameters, self.__holdout_parameters = data_parameters, holdout_parameters
        self.__models = []
        self.__alphas = alphas
        self.__statistical_tests_scores = {}
        self.__balance_type = balance_type
        self.__defined_algorithms = defined_algorithms

    def execute(self):
        dataset, labels = pipeline((self.__data_parameters, self.__holdout_parameters[-1]))
        for algorithm in self.__defined_algorithms:
            for name, hyperparameters in self.__defined_algorithms.get(algorithm):
                model = Model(algorithm, name, False if type(hyperparameters) == dict else True)
                model.build(hyperparameters)
                self.__models.append(model)
        for holdout, data in enumerate(get_holdouts(dataset, labels, self.__holdout_parameters, self.__data_type)):
            training_data, test_data = data
            X_train, y_train = training_data
            X_test, y_test = test_data
            if self.__balance_type and self.__data_type == 'epigenomic':
                X_train, y_train = balance(X_train, y_train, self.__holdout_parameters[-1], self.__balance_type)
            for model in self.__models:
                model.train(training_data)
                y_train_prediction = model.predict(X_train)
                y_test_prediction = model.predict(X_test)
                for metric in metrics:
                    model.test_metrics(metric, (y_train, y_train_prediction))
                for metric in metrics:
                    model.test_metrics(metric, (y_test, y_test_prediction))

    def evaluate(self):
        for alpha in self.__alphas:
            for statistical_test in statistical_tests:
                self.__statistical_tests_scores[statistical_test.__name__] = []
                for metric in metrics:
                    self.__statistical_tests_scores.get(statistical_test.__name__) \
                        .append(test_models(self.__models, statistical_test, metric, alpha))

    def get_models(self) -> List[Model]:
        return self.__models

    def get_best_models(self) -> Dict[str, List[Tuple[Model, float, str, float]]]:
        return self.__statistical_tests_scores

    def print_model_info(self, models: str = 'models'):
        if models == 'models':
            for model in self.__models:
                print_model(model)
        elif models == 'best':
            for statistical_test in statistical_tests:
                for score in self.__statistical_tests_scores.get(statistical_test.__name__):
                    pprint((score[0].get_name(), score[1:]))
        elif models == 'all':
            for model in self.__models:
                print_model(model)
            print('---- Best models ----')
            for statistical_test in statistical_tests:
                for score in self.__statistical_tests_scores.get(statistical_test.__name__):
                    pprint((score[0].get_name(), score[1:]))



