#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    pw = int(kernel.shape[1] / 2)
    ph = int(kernel.shape[0] / 2)
    nw = images.shape[2] - kernel.shape[1] + (2 * pw) + 1
    nh = images.shape[1] - kernel.shape[0] + (2 * ph) + 1
    convolved = np.zeros((images.shape[0], (nh), (nw)))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        for j in range(nw):
            image = imagesp[:, i:(i + kernel.shape[0]),
                            j:(j + kernel.shape[1])]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel, axis=1),
                                        axis=(1, 2))
    return convolved
