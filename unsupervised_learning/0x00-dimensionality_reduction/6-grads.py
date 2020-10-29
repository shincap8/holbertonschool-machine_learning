#!/usr/bin/env python3
"""Function that calculates the gradients of Y"""

import numpy as np


Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Function that calculates the gradients of Y"""
    Q, num = Q_affinities(Y)
    P_Q = P - Q
    dY = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        dY[i, :] = np.sum(np.tile(P_Q[:, i] * num[:, i],
                                  (Y.shape[1], 1)).T * (Y[i, :] - Y), 0)
    return (dY, Q)
