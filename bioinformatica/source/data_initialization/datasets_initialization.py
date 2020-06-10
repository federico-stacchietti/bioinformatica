from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from bioinformatica.source.data_initialization.utils import get_holdouts
import numpy as np
from bioinformatica.source.data_initialization.utils import load_dataset


def get_data(type):
    if type == 'epigenomic':
        return get_epigenomes
    if type == 'sequences':
        return get_sequences


def get_epigenomes(parameters):
    cell_line, window_size, type, n_split, test_size, random_state = parameters
    epigenomes, labels = load_dataset(cell_line, window_size, type)
    for training_indexes, test_indexes in get_holdouts(n_split, test_size, random_state).split(epigenomes, labels):
        yield ((epigenomes.iloc[training_indexes], labels[training_indexes]),
               (epigenomes.iloc[test_indexes], labels[test_indexes]))


def get_sequences(parameters):
    cell_line, genome, window_size, type, n_split, test_size, random_state = parameters
    epigenomes, epigenomes_labels = load_dataset(cell_line, window_size, type)
    bed = epigenomes.reset_index()[epigenomes.index.names]
    sequences, sequence_labels = [data for data in MixedSequence(
                x=BedSequence(genome, bed.iloc[np.arange(len(epigenomes_labels))], batch_size=len(epigenomes_labels)),
                y=epigenomes_labels[np.arange(len(epigenomes_labels))],
                batch_size=len(epigenomes_labels))[0]]
    for training_indexes, test_indexes in get_holdouts(n_split, test_size, random_state)\
            .split(sequences, sequence_labels):
        yield ((sequences[training_indexes], sequence_labels[training_indexes]),
               (sequences[test_indexes], sequence_labels[test_indexes]))
