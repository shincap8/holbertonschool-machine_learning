#!/usr/bin/env python3
"""Function that converts a one-hot matrix into a vector of labels"""


import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels"""
    if len(one_hot) == 0 or type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
