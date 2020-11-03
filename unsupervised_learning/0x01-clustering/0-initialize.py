#!/usr/bin/env python3
"""Function that initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Function that initializes cluster centroids for K-means"""
    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    return np.random.uniform(low, high, size=(k, d))
