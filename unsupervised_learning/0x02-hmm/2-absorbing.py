#!/usr/bin/env python3
"""Function that determines if a markov chain is absorbing"""

import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return False
    ab_state = np.where(np.diag(P) == 1)
    if len(ab_state[0]) == P.shape[0]:
        return True
    if len(ab_state[0]) == 0:
        return False
    B = np.copy(P)
    B = np.delete(np.delete(B, ab_state[0], 0), ab_state[0], 1)
    In = np.identity(B.shape[0])
    try:
        result = np.linalg.inv(In - B)
        return True
    except Exception:
        return False
    """
    row = P[ab_state[0]]
    count = np.sum(row, axis=0)
    for i in range(P.shape[0]):
        check_r = P[i] != 0
        intersection = count * check_r
        if (intersection == 1).any():
            count[i] = 1
    return count.all()"""
