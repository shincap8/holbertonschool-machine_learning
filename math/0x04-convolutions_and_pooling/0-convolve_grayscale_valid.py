#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    nw = images.shape[2] - kernel.shape[1] + 1
    nh = images.shape[1] - kernel.shape[0] + 1
    convolved = np.zeros((images.shape[0], nh, nw))
    for i in range(nh):
        for j in range(nw):
            image = images[:, i:(i + kernel.shape[0]), j:(j + kernel.shape[1])]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
