#!/usr/bin/env python3
"""Function to return the addition of two matrices"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """Function to return the addition of two matrices"""
    add = []
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    for i in range(len(mat1)):
        add.append(list())
        for j in range(len(mat1[i])):
            add[i].append(mat1[i][j] + mat2[i][j])
    return add
