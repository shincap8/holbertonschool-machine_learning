#!/usr/bin/env python3
"""Function that performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images"""
    nw = int(((images.shape[2] - kernel_shape[1]) / stride[1]) + 1)
    nh = int(((images.shape[1] - kernel_shape[0]) / stride[0]) + 1)
    pooled = np.zeros((images.shape[0], nh, nw, images.shape[3]))
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            image = images[:, x:x + kernel_shape[0],
                           y:y + kernel_shape[1], :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(image, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.average(image, axis=(1, 2))
    return pooled
