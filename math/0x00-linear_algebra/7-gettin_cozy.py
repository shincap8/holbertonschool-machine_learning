#!/usr/bin/env python3
"""Function to return two matrices concatenated"""


cat_arrays = __import__('6-howdy_partner').cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """Function to return two matrices concatenated"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        concat = []
        for i in mat1:
            concat.append(list(i))
        for i in mat2:
            concat.append(list(i))
        return concat
    if len(mat1) != len(mat2):
        return None
    concat = []
    for i in range(len(mat1)):
        concat.append(list())
        concat[i] = cat_arrays(mat1[i], mat2[i])
    return concat
