#!/usr/bin/env python3
"""Function that initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Function that initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or type(k) is not int:
        return None
    if len(X.shape) != 2 or k < 0:
        return None
    n, d = X.shape
    if k == 0:
        return None
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    return np.random.uniform(low, high, size=(k, d))
