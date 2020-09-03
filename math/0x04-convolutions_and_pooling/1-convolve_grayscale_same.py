#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    pw = int(kernel.shape[1] / 2)
    ph = int(kernel.shape[0] / 2)
    convolved = np.zeros((images.shape[0], images.shape[1], images.shape[2]))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            image = imagesp[:, i:i + kernel.shape[0],
                            j:j + kernel.shape[1]]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
