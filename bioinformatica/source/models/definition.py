from ..models.libraries import *


def define_models(data_type, nn_input_dimension=None):
    Input_layer = None
    if data_type == 'epigenomic':
        Input_layer = Input(shape=(nn_input_dimension,))
    else:
        Input_layer = Input(shape=(200, 4))
    algorithms = []
    models = {}
    for name, defined_models in algorithms:
        models[name] = []
        for model in defined_models:
            models.get(name).append(model)
    return models


