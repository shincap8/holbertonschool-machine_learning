#!/usr/bin/env python3
"""Function that performs PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X"""
    data = X - np.mean(X, axis=0)
    u, S, v = np.linalg.svd(data)
    W = v[:ndim].T
    return np.matmul(data, W)
