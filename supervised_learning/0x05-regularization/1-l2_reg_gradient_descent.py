#!/usr/bin/env python3
"""Function that updates the weights and biases of a neural
network using gradient descent with L2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural
    network using gradient descent with L2 regularization"""
    dz = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
        L2 = (lambtha / Y.shape[1]) * weights["W{}".format(i)]
        dw = (np.matmul(cache["A{}".format(i - 1)], dz.T) / Y.shape[1]) + L2.T
        dz = np.matmul(weights["W{}".format(
            i)].T, dz) * (cache["A{}".format(
                i - 1)] * (1 - cache["A{}".format(i - 1)]))
        weights["W{}".format(i)] = weights["W{}".format(
            i)] - (alpha * dw).T
        weights["b{}".format(i)] = weights["b{}".format(
            i)] - (alpha * db)
