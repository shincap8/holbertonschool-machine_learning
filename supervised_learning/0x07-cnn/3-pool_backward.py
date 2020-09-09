#!/usr/bin/env python3
"""Function that performs back propagation
over a pooling layer of a neural network"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation
    over a pooling layer of a neural network"""
    dA_prev = np.zeros(A_prev.shape)
    for m in range(dA.shape[0]):
        for i in range(dA.shape[1]):
            x = i * stride[0]
            for j in range(dA.shape[2]):
                y = j * stride[1]
                for k in range(dA.shape[3]):
                    if mode == 'max':
                        A_min = A_prev[m, x:x + kernel_shape[0],
                                       y:y + kernel_shape[1], k]
                        slope = (A_min == np.max(A_min))
                        dA_prev[m, x:x + kernel_shape[0],
                                y:y + kernel_shape[1],
                                k] += slope*dA[m, i, j, k]
                    else:
                        dAmin = dA[m, i, j, k]/kernel_shape[0]/kernel_shape[1]
                        dA_prev[m, x:x + kernel_shape[0],
                                y:y + kernel_shape[1], k] += dAmin
    return dA
