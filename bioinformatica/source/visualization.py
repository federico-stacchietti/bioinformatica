import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

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

    plt.show()
    # plt.savefig(filename)



if __name__ == '__main__':

    #TEST DI ESECUZIONE per binary classification

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    res = pca.transform(X)

    for x in res:
        print(x)

    print("")

    print(res)#res obbiettivo da stampare

    etichette = [0,1,0,1,0,1]

    print (etichette)

    binary_visualization(res, etichette, 'grafico.png')






    # #test per seconda funzione di visualizzazione
    #
    # #{'Name': (0.5774234009526935, 0.4225765990473065),
    # # 'Age': (-0.3243944336109792, 0.6756055663890208),
    # # 'Job': (-0.418395392655304, 0.581604607344696),
    # # 'Date': (-0.5811590033970508, 0.41884099660294916)}
    #
    #
    #
    # #test per istogrammi
    #
    # data2 = [[100000, 1, 2, 3], [20, 21, 19, 18], [1, 1, 1, 200000], [20, 44, 60, 1000], [20, 45, 70, 1020]]
    # df2 = pd.DataFrame(data2, columns=['Name', 'Age', 'Job', 'Date'])
    #
    # # print(df2)
    #
    # labels = [0,1,0,1,0]
    #
    # sdf = {
    #     "abc" : df2,
    #     "bsd" : df2
    # }
    #
    # # for label, content in df2.items():
    # #     print('label:', label)
    # #
    # #     print('content:', content, sep='\n')
    #
    # prova(sdf, labels)




