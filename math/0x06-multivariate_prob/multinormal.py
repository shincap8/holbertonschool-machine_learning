#!/usr/bin/env python3
"""Class that represents a Multivariate Normal distribution"""

import numpy as np


class MultiNormal:
    """Multinormal Class"""

    def __init__(self, data):
        """Class contructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        self.mean = mean
        self.cov = np.dot(data - self.mean, data.T) / (data.shape[1] - 1)
