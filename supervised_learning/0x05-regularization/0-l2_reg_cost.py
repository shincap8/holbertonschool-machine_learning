#!/usr/bin/env python3
"""Function that calculates the cost of a
neural network with L2 regularization"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a
    neural network with L2 regularization"""
    w = 0
    for i in range(1, L + 1):
        w = w + np.linalg.norm(weights['W' + str(i)])
    return (cost + ((lambtha / (2 * m)) * w))
