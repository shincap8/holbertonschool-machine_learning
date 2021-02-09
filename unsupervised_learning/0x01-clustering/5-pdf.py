#!/usr/bin/env python3
"""Function that calculates the probability
density function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """X is a numpy.ndarray of shape (n, d) containing
    the data points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d)
    containing the covariance of the distribution
    You are not allowed to use any loops
    You are not allowed to use the function
    numpy.diag or the method numpy.ndarray.diagonal
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,)
        containing the PDF values for each data point
    All values in P should have a minimum value of 1e-300

    This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized"""
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return None
    if type(S) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(S.shape) != 2:
        return None
    if len(m.shape) != 1:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    N = np.sqrt((2*np.pi)**d * S_det)
    fac = np.einsum('...k,kl,...l->...', X-m, S_inv, X-m)
    P = np.exp(-fac / 2) / N
    P[np.where(P < 1e-300)] = 1e-300
    return P
