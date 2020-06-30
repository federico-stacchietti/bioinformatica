from bioinformatica.source.models.libraries import *
from multiprocessing import cpu_count


def define_models():
    models = {
    }
    defined_models = {}
    for algorithm in models:
        defined_models[algorithm] = []
        for model in models.get(algorithm):
            defined_models.get(algorithm).append(model)
    return defined_models
