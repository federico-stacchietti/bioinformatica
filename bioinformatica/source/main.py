from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.definition import *
from bioinformatica.source.visualizations.visualization import *
from bioinformatica.source.datasets.loader import get_data
from bioinformatica.source.preprocessing.imputation import imputation


'''
Example of an experiment setup:
    experiment_id = 1
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    balance = 'under_sample'
    save_results = False
    dataset_row_reduction = None
    execute_pipeline = True
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance, 
                            save_results, dataset_row_reduction, execute_pipeline)
    experiment.execute()
    experiment.evaluate()
    experiment.print_model_info('all')

'''

'''
Visualization allows to plot many kind of data. Options include PCA visualization, the balancing of the dataset, the top 
    n different tuples, feature correlation, feature distribution and TSNE.
    Experiment results images can be found inside source.barplots folder
    Below you can find how to execute any visualization 

Example of visualization setup:
    
    data_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    data_parameters = ((cell_line, window_size, epigenomic_type), data_type)
    dataset, labels = get_data(data_parameters)
    dataset = imputation(dataset)
    
    make_visualization('experiment_results')
    make_visualization('PCA', dataset, labels, cell_line, epigenomic_type, data_type, PCA_n_components=50)
    make_visualization('balancing', dataset, labels, cell_line, epigenomic_type, data_type)
    make_visualization('top_different_tuples', dataset, labels, cell_line, epigenomic_type, data_type,
                       top_different_tuples=5)
    make_visualization('feature_correlations', dataset, labels, cell_line, epigenomic_type, data_type)
    make_visualization('feature_distribution', dataset, labels, cell_line, epigenomic_type, data_type)
    make_visualization('TSNE', dataset, labels, cell_line, epigenomic_type, data_type,
                       TSNE_n_components=2, PCA_before_TSNE=True, PCA_n_components=75, TSNE_perplexity=40)
'''


if __name__ == '__main__':
    pass

