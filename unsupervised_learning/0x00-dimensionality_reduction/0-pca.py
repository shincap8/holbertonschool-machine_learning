#!/usr/bin/env python3
"""Function that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    U, S, V = np.linalg.svd(X)
    acum = np.cumsum(S)
    dim = []
    for i in range(len(S)):
        if ((acum[i]) / acum[-1]) >= var:
            dim.append(i)
    r = dim[0] + 1
    return V.T[:, :r]
