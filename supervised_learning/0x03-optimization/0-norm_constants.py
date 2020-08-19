#!/usr/bin/env python3
"""Function that calculates the normalization
(standardization) constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization
    (standardization) constants of a matrix"""
    mean = sum(X) / X.shape[0]
    X = X - mean
    desv = (sum(X ** 2) / X.shape[0]) ** 0.5
    return (mean, desv)
