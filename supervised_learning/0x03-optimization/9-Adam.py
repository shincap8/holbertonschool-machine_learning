#!/usr/bin/env python3
"""Function that updates a variable in place
using the Adam optimization algorithm"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place
    using the Adam optimization algorithm"""
    Vd = np.multiply(beta1, v) + np.multiply((1 - beta1), grad)
    Sd = np.multiply(beta2, s) + np.multiply((1 - beta2), grad ** 2)
    Vdc = np.divide(Vd, (1 - (beta1 ** t)))
    Sdc = np.divide(Sd, (1 - (beta2 ** t)))
    var = var - np.multiply(alpha, np.divide(Vdc, ((Sdc ** 0.5) + epsilon)))
    return (var, Vd, Sd)
