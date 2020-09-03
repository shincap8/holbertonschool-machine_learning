#!/usr/bin/env python3
"""Function that performs a valid convolution
on grayscale images with custom padding"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a valid convolution
    on grayscale images with custom padding"""
    if padding == 'same':
        ph = int(((images.shape[2] - 1) * stride[1] +
                  kernel.shape[1] - images.shape[2]) / 2) + 1
        pw = int(((images.shape[1] - 1) * stride[0] +
                  kernel.shape[0] - images.shape[1]) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((images.shape[2] - kernel.shape[1] + (2 * pw)) / stride[1]) + 1)
    nh = int(((images.shape[1] - kernel.shape[0] + (2 * ph)) / stride[0]) + 1)
    convolved = np.zeros((images.shape[0], nh, nw))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            image = imagesp[:, x:x + kernel.shape[0],
                            y:y + kernel.shape[1]]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
