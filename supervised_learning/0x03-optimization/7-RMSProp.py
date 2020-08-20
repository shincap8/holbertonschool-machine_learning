#!/usr/bin/env python3
"""Function that updates a variable using
the RMSProp optimization algorithm"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Function that updates a variable
    using the RMSProp optimization algorithm"""
    Sd = np.multiply(beta2, s) + np.multiply((1 - beta2), grad ** 2)
    var = var - np.multiply(alpha, np.divide(grad, ((Sd ** 0.5) + epsilon)))
    return (var, Sd)
