#!/usr/bin/env python3
"""Function that performs a valid convolution
on grayscale images with custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a valid convolution
    on grayscale images with custom padding"""
    nw = int(images.shape[2] - kernel.shape[1] + (2 * padding[1]) + 1)
    nh = int(images.shape[1] - kernel.shape[0] + (2 * padding[0]) + 1)
    convolved = np.zeros((images.shape[0], nh, nw))
    npad = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        for j in range(nw):
            image = imagesp[:, i:i + kernel.shape[0],
                            j:j + kernel.shape[1]]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
