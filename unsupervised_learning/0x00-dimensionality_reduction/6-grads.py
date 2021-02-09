#!/usr/bin/env python3
"""Function that calculates the gradients of Y"""

import numpy as np


Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Y is a numpy.ndarray of shape (n, ndim)
    containing the low dimensional transformation of X
    P is a numpy.ndarray of shape (n, n)
    containing the P affinities of X
    Do not multiply the gradients by the scalar
    4 as described in the paper’s equation
    Returns: (dY, Q)
        dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
    You may use Q_affinities = __import__('5-Q_affinities').Q_affinities"""
    Q, num = Q_affinities(Y)
    P_Q = P - Q
    dY = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        dY[i, :] = np.sum(np.tile(P_Q[:, i] * num[:, i],
                                  (Y.shape[1], 1)).T * (Y[i, :] - Y), 0)
    return (dY, Q)
