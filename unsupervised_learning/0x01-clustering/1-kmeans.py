#!/usr/bin/env python3
"""Function that performs K-means on a dataset"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the
    maximum number of iterations that should be performed
    If no change in the cluster centroids occurs
    between iterations, your function should return
    Initialize the cluster centroids using a multivariate
    uniform distribution (based on0-initialize.py)
    If a cluster contains no data points during the
    update step, reinitialize its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the
        index of the cluster in C that each data point belongs to"""
    if type(X) is not np.ndarray or type(k) is not int:
        return (None, None)
    if len(X.shape) != 2 or k < 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)
    n, d = X.shape
    if k == 0:
        return (None, None)
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        new_C = np.copy(C)
        for c in range(k):
            if c not in clss:
                new_C[c] = np.random.uniform(low, high)
            else:
                new_C[c] = np.mean(X[clss == c], axis=0)
        if (new_C == C).all():
            return (C, clss)
        else:
            C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return (C, clss)
