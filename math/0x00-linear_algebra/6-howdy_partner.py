#!/usr/bin/env python3
"""Function to return two matrices concatenated"""


def cat_arrays(arr1, arr2):
    """Function to return two matrices concatenated"""
    concat = list(arr1)
    for i in arr2:
        concat.append(i)
    return concat
