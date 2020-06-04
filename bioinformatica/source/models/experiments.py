from bioinformatica.source.data_initialization.datasets_initialization import get_data

from sklearn import svm as SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.layers import LSTM

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
            for training_set, training_labels, test_set, test_labels, hyperparameters in \
                    zip(self.data, self.hyperparameters_list):
                model.fit(training_set, training_labels, hyperparameters)
                self.results.append((model.predict(training_set), model.predict(test_set)))
