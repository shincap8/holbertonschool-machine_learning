#!/usr/bin/env python3
"""Function to return two nmatrices concatenated"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    return np.concatenate((mat1, mat2), axis=axis)
