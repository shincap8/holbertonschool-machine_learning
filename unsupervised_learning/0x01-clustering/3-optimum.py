#!/usr/bin/env python3
"""Function that tests for the optimum number of clusters by variance"""

import numpy as np
variance = __import__('2-variance').variance
kmeans = __import__('1-kmeans').kmeans


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by variance"""
    if type(X) is not np.ndarray:
        return (None, None)
    if len(X.shape) != 2 or kmin < 0 or kmax < 0:
        return (None, None)
    if iterations <= 0:
        return (None, None)
    results = []
    variances = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k)
        var = variance(X, C)
        results.append(C)
        results.append(clss)
        variances.append(var)
    first = variances[0]
    d_vars = []
    for i in range(len(variances)):
        d_vars.append(first - variances[i])
    return (results, d_vars)
