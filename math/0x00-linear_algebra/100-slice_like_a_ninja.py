#!/usr/bin/env python3
"""slices a matrix along a specific axes"""


def np_slice(matrix, axes={}):
    """slices a matrix along a specific axes"""
    slices = []
    for i in range(max(axes.keys()) + 1):
        if i not in axes:
            slices.append(slice(None, None, None))
        else:
            slices.append(eval("slice{}".format(axes[i])))
    return matrix[tuple(slices)]
