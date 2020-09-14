from bioinformatica.source.preprocessing.decompositions import PCA_function, TSNE_function
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from bioinformatica.source.preprocessing.correlation import *
from barplots import barplots
import os
from pathlib import Path


def visualize_balance(labels: list or np.array, cell_line: str, epigenomic_type: str, dataset_type: str,):
    filename = cell_line + '_' + dataset_type + '_' + epigenomic_type + '_balance.png'
    plot_title = cell_line + ', ' + epigenomic_type + ' class balance'
    ones = np.count_nonzero(labels == 1)
    zeros = np.count_nonzero(labels == 0)

    plt.figure(figsize=(4, 3))
    plt.grid(b=True, axis='both')
    plt.subplot(111)
    plt.bar(['0', '1'], [zeros, ones])
    plt.title(plot_title)
    path = Path(__file__).parent
    plt.savefig(str(path) + '/dataset_balancing/' + filename, bbox_inches='tight')


def visualize_feature_correlations(dataset: pd.DataFrame, labels: np.array, top: int,
                         p_value: float, cell_line: str, epigenomic_type: str, dataset_type: str):
    features = feature_correlation(dataset)
    filename = cell_line + '_' + dataset_type + '_' + epigenomic_type + '_correlation.png'
    most_correlated = [[int(index) for index in score[0].split(' ')] for score in
                       sorted(features, key=lambda x: x[1], reverse=True)
                       if score[2] < p_value][:top]
    least_correlated = [[int(index) for index in score[0].split(' ')] for score in sorted(features, key=lambda x: x[1])
                        if score[2] < p_value][:top]

    mc_all_indices = list(set([x for y in [(x, y) for x, y in most_correlated] for x in y]))
    lc_all_indices = list(set([x for y in [(x, y) for x, y in least_correlated] for x in y]))

    sns.pairplot(pd.concat([dataset.iloc[:, mc_all_indices], pd.DataFrame(labels)], axis=1), hue=0)

    path = Path(__file__).parent
    plt.savefig(str(path) + '/features_correlations/' + 'MOST_' + filename, bbox_inches='tight')

    sns.pairplot(pd.concat([dataset.iloc[:, lc_all_indices], pd.DataFrame(labels)], axis=1), hue=0)

    plt.savefig(str(path) + '/features_correlations/' + 'least_' + filename, bbox_inches='tight')


def visualize_top_different_tuples(dataset: pd.DataFrame, top_different: int, cell_line: str, dataset_type: str,
                                   epigenomic_type: str):
    filename = cell_line + '_' + dataset_type + '_' + epigenomic_type + '_different_tuples.png'
    dist = euclidean_distances(dataset.T)
    dist = np.triu(dist)
    tuples = list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:top_different]

    fig, axes = plt.subplots(nrows=1, ncols=top_different, figsize=((top_different * 5), 5))

    for (i, j), axis in zip(tuples, axes.flatten()):
        column_i = dataset.columns[i]
        column_j = dataset.columns[j]
        for column in (column_i, column_j):
            head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()
            mask = ((dataset[column] < tail) & (dataset[column] > head)).values
            dataset[column][mask].hist(ax=axis, bins=20, alpha=0.5)
        axis.set_title(f"{column_i} and {column_j}")
    fig.tight_layout()
    path = Path(__file__).parent

    plt.savefig(str(path) + '/top_different_tuples/' + filename)


def visualize_feature_distribution(dataset: pd.DataFrame, labels: np.array, top_n_features: int,
                                   cell_line: str, dataset_type: str, epigenomic_type: str):
    filename = cell_line + '_' + dataset_type + '_' + epigenomic_type + '_feature_distribution.png'

    dist = euclidean_distances(dataset.T)
    most_distance_columns_indices = np.argsort(-np.mean(dist, axis=1).flatten())[:top_n_features]
    columns = dataset.columns[most_distance_columns_indices]

    fig, axes = plt.subplots(nrows=1, ncols=top_n_features, figsize=((top_n_features * 5), 5))

    for column, axis in zip(columns, axes.flatten()):
        head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()

        mask = ((dataset[column] < tail) & (dataset[column] > head)).values

        cleared_x = dataset[column][mask]
        cleared_y = labels.ravel()[mask]

        cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
        cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

        axis.set_title(column)
    fig.tight_layout()
    path = Path(__file__).parent
    plt.savefig(str(path) + '/features_distributions/' + filename)


def make_PCA(filename: str, points: list, labels: list):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    path = Path(__file__).parent
    fig, ax = plt.subplots(nrows=1, ncols=1)
    xs, ys = [x[0] for x in points], [y[1] for y in points]
    ax.scatter(xs, ys, s=1, color=colors[labels])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(str(path) + '/decompositions/PCA_' + filename)


def make_TSNE(filename: str, points: list, labels: list):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    path = Path(__file__).parent
    fig, ax = plt.subplots(nrows=1, ncols=1)
    xs, ys = [x[0] for x in points], [y[1] for y in points]
    ax.scatter(xs, ys, s=1, color=colors[labels])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(str(path) + '/decompositions/TSNE_' + filename)


def visualize_PCA(dataset: pd.DataFrame, labels: np.array, random_state: int, PCA_n_components: int, cell_line: str,
                  epigenomic_type: str, dataset_type: str):
    dataset = PCA_function(dataset, PCA_n_components, random_state)
    make_PCA(cell_line + '_' + epigenomic_type + '_' + dataset_type, dataset, labels)


def visualize_TSNE(dataset: pd.DataFrame, labels: np.array, random_state: int, TSNE_n_components: int,
                       TSNE_perplexity: int, PCA_before_TSNE: bool, PCA_n_components: int, cell_line: str,
                       epigenomic_type: str, dataset_type: str):
    dataset = TSNE_function(dataset, TSNE_n_components, random_state, TSNE_perplexity, PCA_before_TSNE,
                            PCA_n_components)
    make_TSNE(cell_line + '_' + epigenomic_type + '_' + dataset_type, dataset, labels)


def make_barplots(filename: str, results: pd.DataFrame):
    barplots(results,
             groupby=["model", "run_type"],
             show_legend=False,
             height=5,
             orientation="horizontal",
             path='barplots/' + filename + '{feature}.png',
             )


def visualize_experiment_scores():
    path = Path(__file__).parent.parent
    experiment_files = os.listdir(str(path) + '/experiments/results')
    if len(experiment_files) >= 1:
        experiment_files = experiment_files[1:]
        for file in experiment_files:
            print(str(file)[:-4] + '_')
            dataframe = pd.read_csv(str(path) + '/experiments/results/' + file)
            make_barplots(str(file)[:-4] + '_', dataframe)
    else:
        print('No experiment has been made, so any result can\'t be plotted')
