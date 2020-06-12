import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA




def imputation(dataset:pd.DataFrame)-> pd.DataFrame:
    #if is_there_nan(dataset):
    dataset = pd.DataFrame(
        KNNImputer(n_neighbors=5).fit_transform(dataset.values),
        columns=dataset.columns,
        index=dataset.index
    )
    return dataset

def feature_correlation(dataset: pd.DataFrame) -> list:
    scores = []
    for i in range(len(dataset.columns)):
        for j in range(i + 1, len(dataset.columns)):
            if ' '.join([str(x) for x in sorted((i, j))]) not in [score[0] for score in scores]:
                correlation, p_value = pearsonr(dataset.iloc[i].values.ravel(), dataset.iloc[j].values.ravel())
                correlation = np.abs(correlation)
                scores.append((' '.join([str(x) for x in sorted((i, j))]), correlation, p_value))
    return scores

#---------------------------------------

#funziona
def balance_visualization(filename:str, plot_title : str, labels:list, counts : np.array):

    plt.figure(figsize=(4, 3))

    plt.subplot(111)
    plt.bar(labels, counts)

    plt.title(plot_title)

    # plt.show()
    plt.savefig(filename)

 #funziona
def feature_correlations_visualization(filename : str, dataset: pd.DataFrame, scores: list, labels : np.array, top: int, p_value: int):
    most_correlated = [[int(index) for index in score[0].split(' ')] for score in sorted(scores, key=lambda x: x[1], reverse=True)
                       if score[2] < p_value][:top]
    least_correlated = [[int(index) for index in score[0].split(' ')] for score in sorted(scores, key=lambda x: x[1])
                       if score[2] < p_value][:top]

    mc_all_indices = list(set([x for y in [(x, y) for x, y in most_correlated] for x in y]))
    lc_all_indices = list(set([x for y in [(x, y) for x, y in least_correlated] for x in y]))

    #indeces = [(x, y) for x in all_indices for y in all_indices]
    #mc_values = [dataset.iloc[:,i] for i in all_indices]

    #most
    sns.pairplot(pd.concat([dataset.iloc[:, mc_all_indices], pd.DataFrame(labels)], axis = 1), hue=0)

    #plt.show()
    plt.savefig('most_'+filename)

    #least
    sns.pairplot(pd.concat([dataset.iloc[:, lc_all_indices], pd.DataFrame(labels)], axis=1), hue=0)

    plt.savefig('least_'+filename)






#funziona
def feature_distribution_visualization(filename : str, dataset : pd.DataFrame, labels : np.array, top_number : int):
    dist = euclidean_distances(dataset.T)
    most_distance_columns_indices = np.argsort(-np.mean(dist, axis=1).flatten())[:top_number]
    columns = dataset.columns[most_distance_columns_indices]

    fig, axes = plt.subplots(nrows=1, ncols=top_number, figsize=((top_number*5), 5))

    for column, axis in zip(columns, axes.flatten()):
        head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()

        mask = ((dataset[column] < tail) & (dataset[column] > head)).values

        cleared_x = dataset[column][mask]
        cleared_y = labels.ravel()[mask]

        cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
        cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

        axis.set_title(column)
    fig.tight_layout()
    #plt.show()
    plt.savefig(filename)

#funziona
def top_different_tuples_visualization(filename : str, dataset : pd.DataFrame, top_number : int):
    dist = euclidean_distances(dataset.T)
    dist = np.triu(dist)
    tuples = list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:top_number]

    fig, axes = plt.subplots(nrows=1, ncols=top_number, figsize=((top_number*5), 5))

    for (i, j), axis in zip(tuples, axes.flatten()):
        column_i = dataset.columns[i]
        column_j = dataset.columns[j]
        for column in (column_i, column_j):
            head, tail = dataset[column].quantile([0.05, 0.95]).values.ravel()
            mask = ((dataset[column] < tail) & (dataset[column] > head)).values
            dataset[column][mask].hist(ax=axis, bins=20, alpha=0.5)
        axis.set_title(f"{column_i} and {column_j}")
    fig.tight_layout()
    #plt.show()
    plt.savefig(filename)

#funziona
def PCA_TSNE_visualization(filename : str, points : list, labels : list, label_x : str,  label_y :str):

    xs0 = [points[index][0] for index in range(len(labels)) if labels[index]==0]
    ys0 = [points[index][1] for index in range(len(labels)) if labels[index]==0]
    xs1 = [points[index][0] for index in range(len(labels)) if labels[index]==1]
    ys1 = [points[index][1] for index in range(len(labels)) if labels[index]==1]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(xs0, ys0, 'bo', xs1, ys1, 'ro')

    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)

    # plt.show()
    plt.savefig(filename)


if __name__ == '__main__':

    # labels = ['lab0', 'lab1']
    # counts = [100, 1000]
    # balance_visualization('prova.png', 'prova', labels, np.array(counts))



    epigenoma = pd.read_csv('/home/willy/HEK293.csv')
    # epigenoma = epigenoma.head(1000)
    # indici = random.sample(range(0, 200), 100)
    # epigenoma = epigenoma.drop(epigenoma.columns[indici], axis=1)


    # etichette = np.array([])
    etichette_file = pd.read_csv('/home/willy/promoters.bed')
    for region, dataset in etichette_file.items():
        if region == 'HEK293':
            etichette = dataset.values.ravel()

    feature_correlations_visualization('fcv.png', imputation(epigenoma), feature_correlation(imputation(epigenoma)), etichette, 3, 0.05)

    #feature_distribution_visualization('dis.png', imputation(epigenoma), etichette, 5)

    #top_different_tuples_visualization('top.png', imputation(epigenoma), 5)




    # TEST DI ESECUZIONE per binary classification

    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # pca = PCA(n_components=2)
    # pca.fit(X)
    # res = pca.transform(X)
    #
    # for x in res:
    #     print(x)
    #
    # print("")
    #
    # print(res)  # res obbiettivo da stampare
    #
    # etichette = [0, 1, 0, 1, 0, 1]
    #
    # print(etichette)
    #
    # PCA_TSNE_visualization('defcvg.png', res, etichette, 'x', 'y')



