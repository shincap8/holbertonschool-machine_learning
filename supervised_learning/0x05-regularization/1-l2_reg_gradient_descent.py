#!/usr/bin/env python3
"""Function that updates the weights and biases of a neural
network using gradient descent with L2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization"""
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        L2 = (lambtha / Y.shape[1]) * weights["W" + str(i)]
        db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
        dw = (np.matmul(dz, cache["A" + str(i - 1)].T) / Y.shape[1]) + L2
        dz = np.matmul(weights["W" + str(
            i)].T, dz) * (1 - (cache["A" + str(
                i - 1)] ** 2))
        weights["b" + str(i)] = weights["b" + str(
            i)] - (alpha * db)
        weights["W" + str(i)] = weights["W" + str(
            i)] - (alpha * dw)
