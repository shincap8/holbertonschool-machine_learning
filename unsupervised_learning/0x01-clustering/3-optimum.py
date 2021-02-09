#!/usr/bin/env python3
"""Function that tests for the optimum number of clusters by variance"""

import numpy as np
variance = __import__('2-variance').variance
kmeans = __import__('1-kmeans').kmeans


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
    number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum
    number of clusters to check for (inclusive)
    iterations is a positive integer containing
    the maximum number of iterations for K-means
    This function should analyze at least 2 different cluster sizes
    You should use:
        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance
    You may use at most 2 loops
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs
        of K-means for each cluster size
        d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size"""
    if type(X) is not np.ndarray:
        return (None, None)
    if type(kmin) is not int:
        return (None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None)
    if kmax is None:
        kmax = X.shape[0]
    if len(X.shape) != 2 or kmin < 1:
        return (None, None)
    if kmax is not None and kmax <= kmin:
        return (None, None)
    if type(iterations) is not int:
        return (None, None)
    if iterations <= 0:
        return (None, None)
    results = []
    variances = []
    k = kmin
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k)
        var = variance(X, C)
        results.append((C, clss))
        variances.append(var)
    first = variances[0]
    d_vars = []
    for i in range(len(variances)):
        d_vars.append(first - variances[i])
    return (results, d_vars)
