#!/usr/bin/env python3
"""Function that tests for the optimum number of clusters by variance"""

import numpy as np
variance = __import__('2-variance').variance
kmeans = __import__('1-kmeans').kmeans


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by variance"""
    if type(X) is not np.ndarray:
        return (None, None)
    if type(kmin) is not int:
        return (None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None)
    if kmax is None:
        kmax = X.shape[0] - 1
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
        results.append(C)
        results.append(clss)
        variances.append(var)
    first = variances[0]
    d_vars = []
    for i in range(len(variances)):
        d_vars.append(first - variances[i])
    return (results, d_vars)
