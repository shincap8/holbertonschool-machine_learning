#!/usr/bin/env python3
"""Function to return the elementwise of two nmatrices"""


def np_elementwise(mat1, mat2):
    """Function to return the elementwise of two nmatrices"""
    tuple = (mat1+mat2, mat1-mat2, mat1*mat2, mat1/mat2)
    return (tuple)
