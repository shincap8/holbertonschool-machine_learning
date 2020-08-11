#!/usr/bin/env python3
"""Function that converts a numeric label vector into a one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    if len(Y) == 0 or Y.max() >= classes:
        return None
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    hot = np.zeros([classes, Y.shape[0]])
    for i, num in enumerate(Y):
        hot[num][i] = 1
    return hot
