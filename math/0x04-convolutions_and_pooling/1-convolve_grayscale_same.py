#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    nw = images.shape[2] - kernel.shape[1] + 1
    nh = images.shape[1] - kernel.shape[0] + 1
    pw = int(kernel.shape[1] / 2)
    ph = int(kernel.shape[0] / 2)
    convolved = np.zeros((images.shape[0], (nh + (2*ph)), (nw + (2*pw))))
    npad = ((0, 0), (1, ph), (1, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh + (2*ph)):
        for j in range(nw + (2*pw)):
            image = imagesp[:, i:(i + kernel.shape[0]),
                            j:(j + kernel.shape[1])]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
