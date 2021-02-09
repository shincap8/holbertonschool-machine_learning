#!/usr/bin/env python3
"""Function that calculates the expectation
step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the
    posterior probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
        containing the updated priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing
        the updated centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing
        the updated covariance matrices for each cluster"""
    if type(X) is not np.ndarray or type(g) is not np.ndarray:
        return (None, None, None)
    if len(X.shape) != 2 or len(g.shape) != 2:
        return (None, None, None)
    if X.shape[0] != g.shape[1]:
        return (None, None, None)
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)
    n, d = X.shape
    k, _ = g.shape
    pi = g.sum(axis=1) / n
    m = np.dot(g, X) / g.sum(1)[:, None]
    S = np.zeros([k, d, d])
    for i in range(k):
        ys = X - m[i, :]
        S[i] = (g[i, :, None, None] *
                np.matmul(ys[:, :, None], ys[:, None, :])).sum(axis=0)
    S = S / g.sum(axis=1)[:, None, None]
    return (pi, m, S)
