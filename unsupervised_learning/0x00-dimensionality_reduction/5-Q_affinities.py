#!/usr/bin/env python3
"""Function that calculates the Q affinities"""

import numpy as np


def Q_affinities(Y):
    """Function that calculates the Q affinities"""
    sum_Y = np.sum(np.square(Y), 1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    return (Q, num)
