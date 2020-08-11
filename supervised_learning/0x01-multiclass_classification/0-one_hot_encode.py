#!/usr/bin/env python3
"""Function that converts a numeric label vector into a one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    if len(Y.shape) == 1 and len(classes) > 2 and type(classes) is int:
        hot = np.zeros([classes, Y.shape[0]])
        for i in range(len(hot)):
            for j in range(len(Y)):
                if i == Y[j]:
                    hot[i][j] = 1
        return hot
    return None
