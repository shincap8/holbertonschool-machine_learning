#!/usr/bin/env python3
"""Function that performs the expectation maximization for a GMM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Function that performs the expectation maximization for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)
    if type(k) is not int or type(iterations) is not int:
        return (None, None, None, None, None)
    if k <= 0 or iterations <= 0:
        return (None, None, None, None, None)
    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None, None)
    n, d = X.shape
    pi, m, S = initialize(X, k)
    ll_old = 0
    for i in range(iterations + 1):
        g, ll = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        if np.abs(ll_old - ll) < tol:
            break
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {:.5f}'.format(i, ll))
        ll_old = ll
    if verbose:
        print('Log Likelihood after {} iterations: {:.5f}'.format(i, ll))
    return (pi, m, S, g, ll)
