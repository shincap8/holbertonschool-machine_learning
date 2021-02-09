#!/usr/bin/env python3
"""Function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
    number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum
    number of clusters to check for (inclusive)
        If kmax is None, kmax should be set to
        the maximum number of clusters possible
    iterations is a positive integer containing the
    maximum number of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM algorithm
    verbose is a boolean that determines if the EM
    algorithm should print information to the standard output
    You should use expectation_maximization =
    __import__('7-EM').expectation_maximization
    You may use at most 1 loop
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
            pi is a numpy.ndarray of shape (k,) containing the
            cluster priors for the best number of clusters
            m is a numpy.ndarray of shape (k, d) containing the
            centroid means for the best number of clusters
            S is a numpy.ndarray of shape (k, d, d) containing the
            covariance matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing
        the log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing
        the BIC value for each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
            p is the number of parameters required for the model
            n is the number of data points used to create the model
            l is the log likelihood of the model"""
    if type(X) is not np.ndarray or type(kmin) is not int:
        return (None, None, None, None)
    if len(X.shape) != 2:
        return (None, None, None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None, None, None)
    if type(kmax) is not int or kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin <= 0 or kmin >= X.shape[0] or kmin >= kmax:
        return (None, None, None, None)
    if type(tol) is not float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None)
    n, d = X.shape
    ki = []
    li = []
    bi = []
    tup = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations,
                                                   tol, verbose)
        p = (d * k) + (k * d * (d + 1) / 2) + k - 1
        li.append(ll)
        ki.append(k)
        tup.append((pi, m, S))
        BIC = p * np.log(n) - 2 * ll
        bi.append(BIC)
    ll = np.array(li)
    b = np.array(bi)
    top = np.argmin(b)
    return (ki[top], tup[top], ll, b)
