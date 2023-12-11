import rff
import torch
import numpy as np
from sklearn.decomposition import PCA


def pca(x):
    pca = PCA(n_components=10)
    pca.fit(x)
    x_new = pca.transform(x)
    return x_new, pca


def inverse_pca(pca, x):
    return pca.inverse_transform(x)

 
if __name__ == '__main__':
    pass