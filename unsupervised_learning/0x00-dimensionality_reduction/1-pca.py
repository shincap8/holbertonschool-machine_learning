#!/usr/bin/env python3
"""Function that performs PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset"""
    data = X - np.mean(X, axis=0)
    u, S, v = np.linalg.svd(data)
    W = v[:ndim].T
    return np.matmul(data, W)
