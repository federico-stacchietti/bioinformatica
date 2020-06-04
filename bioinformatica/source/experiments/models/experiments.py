from bioinformatica.source.data_initialization.datasets_initialization import get_data

from sklearn.metrics import balanced_accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

from tensorflow.keras.metrics import AUC


class Experiment:
    def __init__(self, data_type, experiment_params):
        self.data_type = data_type
        self.experiment_params = experiment_params
        self.data = get_data(data_type)(experiment_params)
        self.models, self.hyperparameters_list = [], []
        self.results = []

    def train(self):
        for model in self.models:
            pass
