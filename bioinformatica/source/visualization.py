import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#global for test
righe = 2
colonne = 4

colors = np.array([
    "tab:blue",
    "tab:orange",
])

def pca(x:np.ndarray, n_components:int=2)->np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)





def visualization(nrows, ncols, function, xs, ys, titles, colors):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(32, 16))

    for x, y, title, axis in zip(xs, ys, titles, axes.flatten()):
        axis.scatter(*function(x).T, s=1, color=colors[y])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.set_title(f"PCA decomposition - {title}")
    plt.show()

#main
visualization(righe, colonne, pca, xs, ys, titles, colors)
