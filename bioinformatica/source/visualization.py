import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


import pandas as pd


# #funzione per test
# def pearson_test(df:pd.DataFrame, labels: list)->dict:
#     score = {feature: [] for feature in df}
#     for feature, x in df.items():
#         score[feature] = pearsonr(x.values.ravel(), labels)
#
#     return score

def binary_visualization(points, labels, filename):

    #si può ottimizzare?
    xs0 = []
    ys0 = []
    xs1 = []
    ys1 = []

    for index in range(len(labels)):

        if labels[index]==0:
            xs0.append(points[index][0])
            ys0.append(points[index][1])
        else:
            xs1.append(points[index][0])
            ys1.append(points[index][1])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(xs0, ys0, 'bo', xs1, ys1, 'ro')

    ax1.set_ylabel('y')
    ax1.set_xlabel('x')

    #plt.show()
    plt.savefig(filename)





def get_top_most_different(dist, n:int):
    return np.argsort(-np.mean(dist, axis=1).flatten())[:n]



def prova(df, labels):

    top_number = 3

    # print(df)

    for region, x in df.items():
        # print(region)
        # print(x)
        print(x)

        dist = euclidean_distances(x.T)

        print(dist)

        most_distance_columns_indices = get_top_most_different(dist, top_number)

        print(most_distance_columns_indices)
        columns = x.columns[most_distance_columns_indices]

        print(columns)

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))


        # print(f"Top {top_number} different features from {region}.")

        for column, axis in zip(columns, axes.flatten()):
            head, tail = x[column].quantile([0.05, 0.95]).values.ravel()

            # print(head)
            #
            # print(tail)

            mask = ((x[column] < tail) & (x[column] > head)).values

            cleared_x = x[column][mask]



            new_labels = []
            for index in range(len(mask)):
                if mask[index]==True:
                    new_labels.append(labels[index])

            cleared_y = new_labels

            print(cleared_x)
            print(cleared_y)

            print(cleared_x[cleared_y == 0])

            # cleared_x[cleared_y == 0].hist(ax=axis, bins=20)
            # cleared_x[cleared_y == 1].hist(ax=axis, bins=20)

            axis.set_title(column)
        # fig.tight_layout()
        # plt.show()



if __name__ == '__main__':

    # #TEST DI ESECUZIONE per binary classification
    #
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
    # print(res)#res obbiettivo da stampare
    #
    # etichette = [0,1,0,1,0,1]
    #
    # print (etichette)
    #
    # binary_visualization(res, etichette, 'grafico.png')






    #test per seconda funzione di visualizzazione

    #{'Name': (0.5774234009526935, 0.4225765990473065),
    # 'Age': (-0.3243944336109792, 0.6756055663890208),
    # 'Job': (-0.418395392655304, 0.581604607344696),
    # 'Date': (-0.5811590033970508, 0.41884099660294916)}



    #test per istogrammi

    data2 = [[100000, 1, 2, 3], [20, 21, 19, 18], [1, 1, 1, 200000], [20, 44, 60, 1000], [20, 45, 70, 1020]]
    df2 = pd.DataFrame(data2, columns=['Name', 'Age', 'Job', 'Date'])

    # print(df2)

    labels = [0,1,0,1,0]

    sdf = {
        "abc" : df2,
        "bsd" : df2
    }

    # for label, content in df2.items():
    #     print('label:', label)
    #
    #     print('content:', content, sep='\n')

    prova(sdf, labels)




