#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    x = matrix
    while type(x) is list:
        shape.append(len(x))
        x = x[0]
    return shape