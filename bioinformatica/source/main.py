from bioinformatica.source.experiments.builder import Experiment
from bioinformatica.source.experiments.definition import *
from bioinformatica.source.visualizations.utils import *
from bioinformatica.source.datasets.loader import get_data
from bioinformatica.source.preprocessing.imputation import imputation


'''
Preprocessing pipeline parameters can be set in preprocessing/pipeline.py file. Also, there can be decided which step
to perform and you can add your own custom functions

Example of an experiment setup (see experiments/builder.py for a detailed description of parameters):
    experiment_id = 1
    dataset_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 10, 0.2, 1
    balance = 'under_sample'
    save_results = True
    execute_pipeline = False
    defined_algorithms = define_models()
    holdout_parameters = (n_split, test_size, random_state)
    data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    alphas = [0.05]
    experiment = Experiment(experiment_id, data_parameters, holdout_parameters, alphas, defined_algorithms, balance, 
                            save_results, execute_pipeline)
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
    
    dataset_type = 'epigenomic'
    cell_line, window_size, epigenomic_type = 'K562', 200, 'enhancers'
    n_split, test_size, random_state = 1, 0.2, 1
    data_parameters = ((cell_line, window_size, epigenomic_type), dataset_type)
    dataset, labels = get_data(data_parameters)
    dataset = imputation(dataset)
    
    visualize_experiment_scores()
    visualize_balance(labels, cell_line, epigenomic_type, dataset_type)
    visualize_top_different_tuples(dataset, top_different, cell_line, dataset_type, epigenomic_type)
    visualize_feature_distribution(dataset, labels, 5, cell_line, epigenomic_type, dataset_type)
    visualize_feature_correlations(dataset, labels, top_n_features, 0.95, cell_line, epigenomic_type, dataset_type)
    visualize_PCA(dataset, labels, random_state, PCA_n_components, cell_line, epigenomic_type, dataset_type)
    visualize_TSNE(dataset, labels, random_state, TSNE_n_components, TSNE_perplexity, PCA_before_TSNE, PCA_n_components,
                   cell_line, epigenomic_type, dataset_type)
'''


if __name__ == '__main__':
    pass
