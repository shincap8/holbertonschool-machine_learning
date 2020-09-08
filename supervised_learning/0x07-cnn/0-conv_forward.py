#!/usr/bin/env python3
"""Function that performs forward propagation
over a convolutional layer of a neural network"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation
    over a convolutional layer of a neural network"""
    if padding == 'same':
        ph = int(((A_prev.shape[1] - 1) * stride[0] +
                  W.shape[0] - A_prev.shape[1]) / 2) + 1
        pw = int(((A_prev.shape[2] - 1) * stride[1] +
                  W.shape[1] - A_prev.shape[2]) / 2) + 1
    else:
        ph = 0
        pw = 0
    nh = int(((A_prev.shape[1] - W.shape[0] + (2 * ph)) / stride[0]) + 1)
    nw = int(((A_prev.shape[2] - W.shape[1] + (2 * pw)) / stride[1]) + 1)
    convolved = np.zeros((A_prev.shape[0], nh, nw, W.shape[3]))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_pad = np.pad(A_prev, pad_width=npad,
                   mode='constant', constant_values=0)
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            for k in range(W.shape[3]):
                A = A_pad[:, x:x + W.shape[0],
                          y:y + W.shape[1], :]
                kernel = W[:, :, :, k]
                convolved[:, i, j, k] = np.sum(np.multiply(A, kernel),
                                               axis=(1, 2, 3))
    convolved = convolved + b
    convolved = activation(convolved)
    return convolved
