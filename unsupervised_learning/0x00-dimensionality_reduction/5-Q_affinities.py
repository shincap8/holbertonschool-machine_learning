#!/usr/bin/env python3
"""Function that calculates the Q affinities"""

import numpy as np


def Q_affinities(Y):
    """Y is a numpy.ndarray of shape (n, ndim) containing
    the low dimensional transformation of X
        n is the number of points
        ndim is the new dimensional representation of X
    Returns: Q, num
        Q is a numpy.ndarray of shape (n, n) containing the Q affinities
        num is a numpy.ndarray of shape (n, n)
        containing the numerator of the Q affinities"""
    sumY = np.sum(np.square(Y), 1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sumY).T, sumY))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / np.sum(num)
    return (Q, num)
