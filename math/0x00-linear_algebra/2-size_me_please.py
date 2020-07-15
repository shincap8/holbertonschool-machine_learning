#!/usr/bin/env python3
"""Some comment"""


def matrix_shape(matrix):
    """Function to return the shape of the matrix"""
    shape = []
    x = matrix
    while type(x) is list:
        shape.append(len(x))
        x = x[0]
    return shape
