#!/usr/bin/env python3
"""Function that converts a one-hot matrix into a vector of labels"""


import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels"""
    try:
        one_hot.shape[1]
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
