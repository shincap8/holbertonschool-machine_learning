#!/usr/bin/env python3
"""class DeepNeuralNetwork that defines a deep neural
network performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class"""

    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.weights["W1"] = np.random.randn(
            layers[0], nx)*np.sqrt(2/nx)
        self.weights["b1"] = np.zeros([layers[0], 1], dtype=float)
        if layers[0] < 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if i < 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W{}".format(
                i + 1)] = np.random.randn(layers[i],
                                          layers[i-1])*np.sqrt(2/layers[i-1])
            self.weights["b{}".format(
                i + 1)] = np.zeros([layers[i], 1], dtype=float)
