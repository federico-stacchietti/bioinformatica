import pandas as pd
import numpy as np
from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence

def load_dataset(cell_line, window_size, type):
    epigenomes, labels = load_epigenomes(
        cell_line=cell_line,
        dataset='fantom',
        regions=type,
        window_size=window_size
    )
    labels = labels.values.ravel()
    return epigenomes, labels

