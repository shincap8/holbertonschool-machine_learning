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
        self.cov = np.matmul(data - self.mean, data.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """public instance method that calculates the PDF at a data point"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if len(data.shape) != 2:
            str = 'x must have the shape ({}, 1)'.format(d)
            raise ValueError(str)
        if x.shape[0] != d or x.shape[1] != 1:
            str = 'x must have the shape ({}, 1)'.format(d)
            raise ValueError(str)
        const = 1 / np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(self.cov)))
        n_dev = -(x - self.mean).T
        ins = np.matmul(n_dev,  np.linalg.inv(self.cov))
        half_dev = (x - self.mean) / 2
        out = np.matmul(ins, half_dev)
        exp = np.exp(out)
        pdf = const * exp
        pdf = pdf.reshape(-1)[0]
        return pdf
