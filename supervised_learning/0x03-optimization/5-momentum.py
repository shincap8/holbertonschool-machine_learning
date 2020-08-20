#!/usr/bin/env python3
"""Function that updates a variable using the gradient
descent with momentum optimization algorithm"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the gradient
    descent with momentum optimization algorithm"""
    Vd = np.multiply(beta1, v) + np.multiply((1 - beta1), grad)
    var = var - np.multiply(alpha, Vd)
    return (var, Vd)
