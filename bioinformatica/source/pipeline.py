from ucsc_genomes_downloader import Genome

from bioinformatica.source.data_initialization.datasets_initialization import get_data
from bioinformatica.source.models_builder.experiments_builder.builder import *
import pandas as pd
import numpy as np
import time

from bioinformatica.source.preprocessing.correlation import filter_uncorrelated, filter_correlated_features
from bioinformatica.source.preprocessing.elaboration import balance, robust_zscoring, drop_constant_features
from bioinformatica.source.preprocessing.feature_selection import boostaroota_filter
from bioinformatica.source.preprocessing.imputation import nan_check, nan_filter, imputation


def pipeline(data_parameters):

    load_parameters, data_type = data_parameters
    cell_line, window_size, epigenomes_type = load_parameters


    #data_retrival

    dataset, labels = get_data(data_parameters)

    #data elaboration

    if data_type == 'epigenomic':


    #visualization?



if __name__ == '__main__':
    random_state = 42
    p_value_threshold, min_correlation, good_correlation = 0.01, 0.05, 0.95

    # epigenoma = pd.read_csv('/home/flavio/Downloads/HEK293.csv')
    # # epigenoma = epigenoma.head(1000)
    # # indici = random.sample(range(0, 200), 100)
    # # epigenoma = epigenoma.drop(epigenoma.columns[indici], axis=1)
    #
    # # etichette = np.array([])
    # etichette_file = pd.read_csv('/home/flavio/Downloads/promoters.bed')
    # for region, x in etichette_file.items():
    #     if region == 'HEK293':
    #         etichette = x.values.ravel()

    data_type = 'epigenomic'
    genome = Genome('hg19')
    cell_line, window_size, typez, n_split, test_size = 'HEK293', 200, 'enhancers', 1, 0.001
    experiment_params = (cell_line, window_size, typez, n_split, test_size, random_state)
    seq_params = (cell_line, genome, window_size, typez, n_split, test_size, random_state)

    sequences = get_sequences(seq_params)

    print(sequences)

    #print(sequences.shape[-1])

    for element in sequences:
        if(np.sum(element)!=1):
            print('trovato')
    # i, j = 0, 0
    # for x in sequences:
    #     if x.shape[-1] == 200:
    #         print(x.shape[-1])
    #         i += 1
    #     for y in x:
    #         if y.shape[-1] == 4:
    #             # print(y.shape[-1])
    #             j += 1
    # print(len(sequences) * i * j == 99909 * 200 * 4)

    # dataset, sadsdsadasd = get_epigenomes(experiment_params).__next__()
    # epigenoma, etichette = dataset
    # epigenoma.reset_index(drop=True, inplace=True)
    # # a = 0
    # # a = dataset
    # # print(type(dataset.values))
    #
    #
    # if nan_check(epigenoma):
    #     epigenoma, etichette = nan_filter(epigenoma, etichette)
    #     epigenoma = imputation(epigenoma)
    #
    # # elaboration
    # epigenoma_bilanciato, etichette_bilanciate = balance(epigenoma, etichette, random_state)
    # epigenoma_zscoring = robust_zscoring(drop_constant_features(epigenoma_bilanciato))
    #
    # # correlation
    # epigenoma_correlati_output = filter_uncorrelated(epigenoma_zscoring, etichette_bilanciate, p_value_threshold, min_correlation)
    # epigenoma_feature_correlate = filter_correlated_features(epigenoma_correlati_output, p_value_threshold, good_correlation)
    #
    # # feature_selection
    # epigenoma_boosted = boostaroota_filter(epigenoma_feature_correlate, etichette_bilanciate)
    #
    # print(epigenoma_boosted)