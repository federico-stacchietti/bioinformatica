from multiprocessing import cpu_count

from ..models.libraries import *


def define_models(data_type, nn_input_dimension=None):
    Input_layer = None
    if data_type == 'epigenomic':
        Input_layer = Input(shape=(nn_input_dimension, ))
    else:
        Input_layer = Input(shape=(200, 4))

    DecTrees = [
        dict(criterion="gini",
             splitter='best',
             max_depth=None,
             min_samples_split=2,
             min_samples_leaf=1,
             min_weight_fraction_leaf=0.0,
             max_features=None,
             random_state=42,
             max_leaf_nodes=None,
             min_impurity_decrease=0.0)
        # dict(criterion="gini",
        #      max_depth=5,
        #      random_state=42,
        #      class_weight="balanced"),
        # dict(criterion= "gini",
        #      max_depth=10,
        #      random_state=42,
        #      class_weight="balanced"),
        # dict(criterion="gini",
        #      max_depth=20,
        #      random_state=42,
        #      class_weight="balanced"),
        # dict(criterion="gini",
        #      max_depth=50,
        #      random_state=42,
        #      class_weight="balanced")
    ]

    RandFor = [
        dict(n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fracition_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
             bootstrap=True,
             oob_score=False,
             n_jobs=-1,
             class_weight=None,
            random_state=42)
        # dict(
        #     n_estimators=500,
        #     criterion="gini",
        #     max_depth=10,
        #     random_state=42,
        #     class_weight="balanced",
        #     n_jobs=cpu_count()),
        #
        # dict(
        #     n_estimators=50,
        #     criterion="gini",
        #     random_state=42,
        #     n_jobs=cpu_count()),
        #
        # dict(
        #     n_estimators=100,
        #     criterion="gini",
        #     max_depth=20,
        #     random_state=42,
        #     class_weight="balanced",
        #     n_jobs=cpu_count()),
        #
        # dict(
        #     n_estimators=500,
        #     criterion="gini",
        #     max_depth=20,
        #     random_state=42,
        #     class_weight="balanced",
        #     n_jobs=cpu_count()),
    ]

    SGDs = [
        dict(loss="hinge",
             penalty="l2",
             alpha=0.0001,
             l1_ratio=0.15,
             fit_intercept=True,
             max_iter=1000,
             tol=1e-3,
             shuffle=True,
             epsilon=0.1,
             n_jobs=-1,
             random_state=42,
             learning_rate='optimal',
             eta0=0.0,
             power_t=0.5,
             early_stopping=False,
             validation_fraction=0.1,
             n_iter_no_change=5,
             class_weight=None,
             warm_start=False)
        # dict(loss="hinge", penalty="l2", max_iter=1000),
        # dict(loss="hinge", penalty="l2", max_iter=2000),
        # dict(loss="hinge", penalty="l2", max_iter=50),
    ]

    AdaBoostClassifiers = [
        dict(base_estimator=None,
             n_estimators=50,
             learning_rate=1,
             algorithm='SAMME.R',
             random_state=42)
    ]

    algorithms = [('DecTree', DecTrees), ('RandForest', RandFor), ('SGD', SGDs), ('AdaBoost', AdaBoostClassifiers)]
    models = {}
    for name, defined_models in algorithms:
        models[name] = []
        for model in defined_models:
            models.get(name).append(model)
    return models

