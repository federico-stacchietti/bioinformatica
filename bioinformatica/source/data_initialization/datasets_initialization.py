from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence
from bioinformatica.source.data_initialization.utils import holdouts
import numpy as np
from bioinformatica.source.data_initialization.utils import load_dataset
from ucsc_genomes_downloader import Genome


def get_data(parameters):
    data_parameters, data_type = parameters
    if data_type == 'epigenomic':
        return load_dataset(data_parameters)
    if data_type == 'sequences':
        epigenomes, labels = load_dataset(data_parameters)
        genome = Genome('hg19')
        bed = epigenomes.reset_index()[epigenomes.index.names]
        batch_size = len(labels)
        return [data for data in MixedSequence(
                x=BedSequence(genome, bed.iloc[np.arange(batch_size)], batch_size=batch_size),
                y=labels[np.arange(batch_size)],
                batch_size=batch_size)[0]]


def get_holdouts(dataset, labels, holdout_parameters):
    for training_indexes, test_indexes in holdouts(holdout_parameters).split(dataset, labels):
        yield ((dataset.iloc[training_indexes], labels[training_indexes]),
               (dataset.iloc[test_indexes], labels[test_indexes]))
