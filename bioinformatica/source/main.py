from bioinformatica.source.models_builder.expereiments_builder.experiments import *

from sklearn.tree import DecisionTreeClassifier as tr

if __name__ == '__main__':
    print()
    # models = ['DecTree', 'NN']
    # data_type = 'epigenomic'
    # cell_line = 'K562'
    # window_size = 200
    # type = 'enhancers'
    # n_split = 1
    # test_size = .2
    # random_state = 1
    # experiment_params = (cell_line, window_size, type, n_split, test_size, random_state)
    # experiment = Experiment(models, data_type, experiment_params)
    # experiment.execute()

    clf = tr(dict(criterion="gini",
        max_depth=50,
        random_state=42,
        class_weight="balanced"))
    a = 0


