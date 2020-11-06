#!/usr/bin/env python3
"""Function that calculates the expectation
step in the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation
    step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return (None, None)
    if type(S) is not np.ndarray or type(pi) is not np.ndarray:
        return (None, None)
    if len(X.shape) != 2 or len(S.shape) != 3:
        return (None, None)
    if len(pi.shape) != 1 or len(m.shape) != 2:
        return (None, None)
    n, d = X.shape
    k = pi.shape[0]
    g = np.zeros([k, n])
    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P
    ll = np.sum(np.log(g.sum(axis=0)))
    g = g / g.sum(axis=0)
    return (g, ll)
