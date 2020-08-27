#!/usr/bin/env python3
"""Function that updates the weights of a neural network
with Dropout regularization using gradient descent"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network
    with Dropout regularization using gradient descent"""
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = (np.matmul(cache["A" + str(i - 1)], dz.T) / Y.shape[1])
        db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
        if i - 1 > 0:
            dz = np.matmul(weights["W" + str(
                i)].T, dz) * (1 - (cache["A" + str(
                    i - 1)] ** 2)) * (cache["D" + str(i - 1)] / keep_prob)
        weights["W" + str(i)] = weights["W" + str(
            i)] - (alpha * dw).T
        weights["b" + str(i)] = weights["b" + str(
            i)] - (alpha * db)
