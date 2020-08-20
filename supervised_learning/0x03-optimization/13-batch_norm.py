#!/usr/bin/env python3
"""Function that normalizes an unactivated output
of a neural network using batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output
    of a neural network using batch normalization"""
    miu = Z.mean(axis=0)
    var = Z.var(axis=0)
    Znom = (Z - miu) / ((var + epsilon) ** 0.5)
    return ((gamma * Znom) + beta)
