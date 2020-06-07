from bioinformatica.source.models_builder.expereiments_builder.experiments import *
from bioinformatica.source.data_initialization.datasets_initialization import get_epigenomes
from bioinformatica.source.models_builder.models.models_libraries import *
from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.layers import Flatten

from sklearn.tree import DecisionTreeClassifier as tr

if __name__ == '__main__':
    print()
    models = ['DecTree', 'SVM', 'NN']
    data_type = 'epigenomic'
    cell_line = 'K562'
    window_size = 200
    type = 'enhancers'
    n_split = 1
    test_size = .2
    random_state = 1
    experiment_params = (cell_line, window_size, type, n_split, test_size, random_state)
    experiment = Experiment(models, data_type, experiment_params)
    experiment.execute()
    experiment.best_by_algorithm()

    # data = get_epigenomes(experiment_params)
    # X1, X2 = data.__next__()
    # X_train, y_train = X1
    # X_test, y_test = X2
    #
    # NN = (([
    #     Input(shape=(298, 1)),
    #     Flatten(),
    #     Dense(1, activation="sigmoid")
    #   ], "Perceptron"),
    #  dict(
    #      optimizer="nadam",
    #      loss="binary_crossentropy"
    #  ),
    #  dict(epochs=1000,
    #       batch_size=1024,
    #       validation_split=0.1,
    #       shuffle=True,
    #       verbose=False,
    #       callbacks=[
    #           EarlyStopping(monitor="val_loss", mode="min", patience=50),
    #       ])
    # )
    #
    # net_p, comp_p, tr_p = NN
    #
    # model = Sequential(*net_p)
    # model.compile(**comp_p)
    # a = model.fit(X_train, y_train, **tr_p)
    #
    # b = 0
    # a = dict(criterion="gini",
    #      max_depth=5,
    #      random_state=42,
    #      class_weight="balanced"),
    # print(' '.join([str(x) for x in a]))


