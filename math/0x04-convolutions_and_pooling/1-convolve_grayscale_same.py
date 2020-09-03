#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    nw = images.shape[2] - kernel.shape[1] + 1
    nh = images.shape[1] - kernel.shape[0] + 1
    p = nw + kernel.shape[1] - images.shape[2]
    convolved = np.zeros((images.shape[0], (nh + (2*p)), (nw + (2*p))))
    imagesp= images
    if p > 0:
        npad = ((0, 0), (1, p), (1, p))
        imagesp = np.pad(imagesp, pad_width=npad, mode='constant', constant_values=0)
    for i in range(nh + (2*p)):
        for j in range(nw + (2*p)):
            image = imagesp[:, i:(i + kernel.shape[0]), j:(j + kernel.shape[1])]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
