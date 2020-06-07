from bioinformatica.source.models_builder.models.libraries import *


def define_models():
    algorithms = []
    models = {}
    for name, defined_models in algorithms:
        models[name] = []
        for model in defined_models:
            models.get(name).append(model)
    return models
