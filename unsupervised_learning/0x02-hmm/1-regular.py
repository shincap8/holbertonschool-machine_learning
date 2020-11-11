#!/usr/bin/env python3
"""Function that determines the steady state
probabilities of a regular markov chain"""

import numpy as np


def regular(P):
    """Function that determines the steady state
    probabilities of a regular markov chain"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return None
    eva, eve = np.linalg.eig(P.T)
    jstat = np.argmin(abs(eva - 1.0))
    stationary = eve[:, jstat].real
    stationary = stationary / stationary.sum()
    if np.min(stationary) <= 0 or np.sum(stationary) != 1:
        return None
    return stationary[np.newaxis, :]
