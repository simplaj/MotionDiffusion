import rff
import torch
import numpy as np
from sklearn.decomposition import PCA


def rff_embedding():
    gs = rff.layers.GaussianEncoding(sigma=10.0, input_size=1, encoded_size=128)
    return gs


def pca(x):
    pca = PCA(n_components=10)
    pca.fit(x)
    x_new = pca.transform(x)
    return x_new, pca


def inverse_pac(pca, x):
    return pca.inverse_transform(x)

 
if __name__ == '__main__':
    pass