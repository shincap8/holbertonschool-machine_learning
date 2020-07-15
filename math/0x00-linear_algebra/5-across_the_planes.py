#!/usr/bin/env python3
"""Function to return the addition of two matrices"""


def add_matrices2D(mat1, mat2):
    """Function to return the addition of two matrices"""
    add = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        add.append(list())
        for j in range(len(mat1[i])):
            add[i].append(mat1[i][j] + mat2[i][j])
    return add
