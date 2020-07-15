#!/usr/bin/env python3
"""Function to return the result of two arrange addition"""


def add_arrays(arr1, arr2):
    """Function to return the result of two arrange addition"""
    add = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])
    return add
