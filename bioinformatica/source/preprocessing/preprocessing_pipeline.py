import pandas as pd
import numpy as np
from bioinformatica.source.preprocessing.data_checking import imputation, balance
from bioinformatica.source.preprocessing.decompositions import data_decomposition


def preprocessing(dataset, labels):
    random_state = 1

    algorithms = ['PCA', 'TSNE']

    dataset = imputation(dataset)
    dataset = balance(dataset, labels, random_state)
    dacompositions = data_decomposition(dataset, algorithms)










