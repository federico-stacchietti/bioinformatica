import pandas as pd
import numpy as np

from bioinformatica.source.data_initialization.utils import load_dataset

from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence


def get_epigenomes(cell_line, window_size, type, holdouts):
    training_sets = []
    test_sets = []
    epigenomes, labels = load_dataset(cell_line, window_size, type)
    for training_indexes, test_indexes in holdouts.split(epigenomes, labels):
        training_sets.append((epigenomes.iloc[training_indexes], labels[training_indexes]))
        test_sets.append((epigenomes.iloc[test_indexes], labels[test_indexes]))
    return zip(training_sets, test_sets)


def get_sequences(cell_line, genome, window_size, type, holdouts):
    training_sets = []
    test_sets = []
    batch_size = 1024
    epigenomes, labels = load_dataset(cell_line, window_size, type)
    bed = epigenomes.reset_index()[epigenomes.index.names]
    for training_indexes, test_indexes in holdouts.split(epigenomes, labels):
        training_sets.append(MixedSequence(
            x=BedSequence(genome, bed.iloc[training_indexes], batch_size=batch_size),
            y=labels[training_indexes],
            batch_size=batch_size
        ))
        test_sets.append(MixedSequence(
            x=BedSequence(genome, bed.iloc[test_indexes], batch_size=batch_size),
            y=labels[test_indexes],
            batch_size=batch_size
        ))
    return zip(training_sets, test_sets)