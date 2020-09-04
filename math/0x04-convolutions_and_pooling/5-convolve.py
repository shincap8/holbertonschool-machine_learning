#!/usr/bin/env python3
"""Function that performs a convolution on images using multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images using multiple kernels"""
    if padding == 'same':
        pw = int(((images.shape[2] - 1) * stride[1] +
                  kernels.shape[1] - images.shape[2]) / 2) + 1
        ph = int(((images.shape[1] - 1) * stride[0] +
                  kernels.shape[0] - images.shape[1]) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((images.shape[2] - kernels.shape[1] + (2 * pw)) / stride[1]) + 1)
    nh = int(((images.shape[1] - kernels.shape[0] + (2 * ph)) / stride[0]) + 1)
    convolved = np.zeros((images.shape[0], nh, nw, kernels.shape[3]))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            for k in range(kernels.shape[3]):
                image = imagesp[:, x:x + kernels.shape[0],
                                y:y + kernels.shape[1], :]
                kernel = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(np.multiply(image, kernel),
                                               axis=(1, 2, 3))
    return convolved
