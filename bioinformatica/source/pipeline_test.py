from multiprocessing import cpu_count
from sklearn.impute import KNNImputer
from utils import load_dataset
from preprocessing import detect_NaN
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
from scipy.stats import entropy
from minepy import MINE
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from copy import deepcopy

# Input:dataframe. Controlla la presenza di feature costanti. Se presenti, le elimina dal dataframe e lo restituisce
def drop_constant_features(dataset:pd.DataFrame)-> pd.DataFrame:
    non_const_features = [feature for feature in dataset.columns if dataset[feature].nunique() > 1]
    return dataset[non_const_features]

#funzia
def is_there_nan(dataset:pd.DataFrame)-> bool:
    for feature, x in dataset.isna().items():
        for element in x:
            if element:
                return True
    return False

#funzia
def detect_nan_in_row(dataset:pd.DataFrame, threshold: int)-> np.array:
    indexes = []
    for index, row in dataset.iterrows():
        if row.count() < threshold:
            indexes.append(index)
    return indexes

# funzia
def nan_filter(dataset:pd.DataFrame, labels:np.array)-> (pd.DataFrame, np.array):
    indexes = detect_nan_in_row(dataset,int((dataset.shape[1]/10)*9))
    dataset = dataset.dropna(axis=0, thresh=int((dataset.shape[1]/10)*9))
    labels = np.delete(arr = labels, obj= indexes)
    dataset = dataset.dropna(axis=1, thresh=int((dataset.shape[0]/10)*9))
    return dataset, labels

#funzia
def imputation(dataset:pd.DataFrame)-> pd.DataFrame:
    if is_there_nan(dataset):
        dataset = pd.DataFrame(
            KNNImputer(n_neighbors=5).fit_transform(dataset.values),
            columns=dataset.columns,
            index=dataset.index
        )
    return dataset

#funzia
def rebalance_classes(dataset:pd.DataFrame, labels:np.array, random_state:int)-> (pd.DataFrame, np.array):
    n_samples = len(dataset)
    max_unbalance = n_samples // 10
    unique, counts = np.unique(labels, return_counts=True)
    minority_label, minority_count = min(zip(unique, counts), key=lambda x: x[1])
    if minority_count < max_unbalance:
        dataset = pd.concat([dataset, resample(dataset.iloc[np.where(labels == minority_label)],
                                               random_state=random_state, n_samples=max_unbalance - minority_count)], axis=0)
        labels = labels + np.full(max_unbalance - minority_count, minority_label)
        np.random.seed(random_state)
        shuffle = np.random.permutation(range(len(labels)))
        dataset, labels = dataset[shuffle], labels[shuffle]
    return dataset, labels

#funzia
def robust_zscoring(dataset:pd.DataFrame)-> pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(dataset.values),
        columns=dataset.columns,
        index=dataset.index
    )

#funzia
# test di Pearson per valutare la correlazione tra feature ed etichette
def pearson_test(dataset:pd.DataFrame, labels:np.array)-> dict:
    score = {feature: [] for feature in dataset}
    for feature, x in dataset.items():
        score[feature] = pearsonr(x.values.ravel(), labels)
    return score

#funzia
# test di Spearman per valutare la correlazione tra feature ed etichette
def spearman_test(dataset:pd.DataFrame, labels:np.array)-> dict:
    score = {feature: [] for feature in dataset}
    for feature, x in dataset.items():
        correlation, p_value = spearmanr(x.values.ravel(), labels)
        score[feature] = correlation, p_value
    return score

#funzia
def uncorrelated_test(dataset:pd.DataFrame, labels:np.array, p_value_threshold:float)-> list:
    pearson_score = pearson_test(dataset,labels)
    spearman_score = spearman_test(dataset,labels)

    pearson_uncorrelated = [key for key,value in pearson_score.items() if value[1]>p_value_threshold]
    spearman_uncorrelated = [key for key,value in spearman_score.items() if value[1]>p_value_threshold]
    return pearson_uncorrelated + [x for x in spearman_uncorrelated if x not in pearson_uncorrelated]

#funzia
# Maximal Information Coefficient
def mic(dataset:pd.DataFrame, labels:np.array)-> dict:
    score = {feature: None for feature in dataset}
    for feature, x in dataset.items():
            mine = MINE()
            mine.compute_score(x.values.ravel(), labels)
            score[feature] = mine.mic()
    return score

#funzia
def filter_uncorrelated(dataset:pd.DataFrame, labels:np.array, p_value_threshold:float, correlation_threshold:float)-> pd.DataFrame:
    uncorrelated_features = uncorrelated_test(dataset, labels, p_value_threshold)
    mic_score = mic(dataset[uncorrelated_features], labels)
    for key in mic_score:
        if mic_score.get(key) >= correlation_threshold:
            uncorrelated_features.remove(key)
    return dataset.drop(columns=uncorrelated_features)

#funzia
def feature_correlation(dataset:pd.DataFrame)->dict:
    score = {}
    for i in range(len(dataset.columns)):
        # for j in dataset.columns[i + 1:]:
        for j in range(i + 1, len(dataset.columns)):
            if ' '.join([str(x) for x in sorted((i, j))]) not in score:
                correlation, p_value = pearsonr(dataset.iloc[i].values.ravel(), dataset.iloc[j].values.ravel())
                correlation = np.abs(correlation)
                score[' '.join([str(x) for x in sorted((i, j))])] = (correlation, p_value)
    return score

def filter_correlated_features(dataset:pd.DataFrame, p_value_threshold:float, correlation_threshold:float)-> pd.DataFrame:
    features = feature_correlation(dataset)

    to_drop = []
    for key in features:
        correlation, p_value = features.get(key)
        first, second = [int(v) for v in key.split(' ')]
        if p_value < p_value_threshold and correlation > correlation_threshold:
            if entropy(dataset.iloc[first]) > entropy(dataset.iloc[second]):
                to_drop.append(second)
            else:
                to_drop.append(first)
    dataset.drop(dataset.columns[list(set(to_drop))], axis=1, inplace=True)
    return dataset

def boruta_filter(dataset:pd.DataFrame, labels:np.array, max_iter:int, p_value_threshold:int, random_state:int,)-> BorutaPy:
    forest = RandomForestClassifier(n_jobs=cpu_count(), class_weight='balanced', max_depth=5)
    boruta_selector = BorutaPy(
        forest,
        n_estimators='auto',
        verbose=2,
        alpha=p_value_threshold, # p_value
        max_iter=max_iter,           #In practice one would run at least 100-200 times
        random_state=random_state
    )
    boruta_selector.fit(dataset.values, labels)#.values.ravel())
    return boruta_selector


if __name__ == '__main__':
    epigenoma = pd.read_csv('/home/flavio/Downloads/HEK293.csv')
    epigenoma = epigenoma.head(1000)
    indici = random.sample(range(0, 200), 100)
    epigenoma = epigenoma.drop(epigenoma.columns[indici], axis=1)


    etichette_file = pd.read_csv('/home/flavio/Downloads/promoters.bed')
    for region, x in etichette_file.items():
        if region == 'HEK293':
            etichette = x.values.ravel()


    #
    # prova = [[1,2,3,np.NaN,5],[1,2,3,np.NaN,5],[1,2,3,np.NaN,5],[np.NaN,np.NaN,np.NaN,np.NaN,np.NaN],[1,2,3,np.NaN,5]]
    # df_prova = pd.DataFrame(prova, columns=['Index','Name', 'Age', 'Job', 'Date'])
    # labels = [0,0,0,0,1]
    #
    # df_prova2 = epigenoma.iloc[np.where(etichette == 0)]
    # df_prova3 = epigenoma.iloc[np.where(etichette == 1)]
    # df_prova3 = df_prova3.head(n=1000)
    # df_prova4 = pd.concat([df_prova2,df_prova3],axis=0)
    # etichette_prova4 = []
    # #print(df_prova4)
    #
    # for i in range(1000):
    #     etichette_prova4.append(0)
    # for i in range(88396):
    #     etichette_prova4.append(1)
    #
    # balanced = rebalance_classes(df_prova4,etichette_prova4)
    #
    # print(len(df_prova4), len(balanced))
    # a = [0, 1, 2, 3, 4, 5, 6]
    # b = resample(a, n_samples=2, replace=False, random_state=0)
    # print(a + b)

    # data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # lab = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # idx = np.random.permutation(len(data))
    # x, y = data[idx], lab[idx]
    # print(x, y)

    prova = [[1, 2, 3, np.NaN, 5], [2, 2, 3, np.NaN, 5], [3, 2, 1, np.NaN, 5], [4, np.NaN, np.NaN, np.NaN, np.NaN], [5, 2, 3, np.NaN, 5]]
    df_prova = pd.DataFrame(prova, columns=['Index','Name', 'Age', 'Job', 'Date'])
    labels = np.array([1, 2, 3, 4, 5])
    #
    # idx = np.random.permutation(5)
    #
    # print(pd.DataFrame(df_prova.iloc[idx], columns=['Index','Name', 'Age', 'Job', 'Date']))
    # labels = labels[idx]
    #
    # print(labels)

    prova = {y:[0 for x in range(10)] for y in range(20,30)}
    df_prova = pd.DataFrame(prova)
    #print(df_prova)
    indexes = [0,1,1,0,1,1,1,1,0,0]
    # print(df_prova)
    # print(nan_filter(df_prova, labels))

    # for column_index in range(20,30):
    #     for index in range(10):
    #         if np.random.randint(0, 5) == 1:
    #             df_prova[column_index][index]=np.NaN
    #print(spearman_test(imputation(epigenoma), etichette))
    # print(filter_uncorrelated(imputation(epigenoma), etichette, 0.01, 0.05))

    # d = {}
    # for i in range(10):
    #     for j in range(10):
    #         if ''.join([str(x) for x in sorted((i, j))]) not in d:
    #             d[''.join([str(x) for x in sorted((i, j))])] = 'a' + str(i) + str(j)
    # # for x in d:
    #     # print(x, d.get(x))
    # print(len(d))

    #print(filter_correlated_features(epigenoma, 0.01, 0.95))

    boruta_filter(df_prova,indexes,300,0.05,0)