#!/usr/bin/env python3
"""Function to return the result of two arrange addition"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1, arr2):
    add = []
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])
    return add
