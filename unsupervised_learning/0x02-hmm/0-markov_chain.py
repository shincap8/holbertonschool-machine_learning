#!/usr/bin/env python3
"""Function that determines the probability of a markov chain
being in a particular state after a specified number of iterations"""

import numpy as np


def markov_chain(P, s, t=1):
    """Function that determines the probability of a markov chain
    being in a particular state after a specified number of iterations"""
    if type(P) is not np.ndarray or type(s) is not np.ndarray:
        return None
    if len(P.shape) != 2 or len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or s.shape[1] != P.shape[0]:
        return None
    if s.shape[0] != 1:
        return None
    sk = np.copy(s)
    for i in range(t):
        sk = np.matmul(sk, P)
    return sk
