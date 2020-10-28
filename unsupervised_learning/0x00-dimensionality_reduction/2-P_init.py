#!/usr/bin/env python3
"""Function that initializes all variables
required to calculate the P affinities in t-SNE"""

import numpy as np


def P_init(X, perplexity):
    """Function that initializes all variables
    required to calculate the P affinities in t-SNE"""
    n = X.shape[0]
    sumX = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX)
    np.fill_diagonal(D, 0.)
    betas = np.ones((n, 1))
    P = np.zeros((n, n))
    sumP = sum(P)
    H = np.log2(perplexity)
    return (D, P, betas, H)
