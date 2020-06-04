import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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




if __name__ == '__main__':

    #TEST DI ESECUZIONE

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

