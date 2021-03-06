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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__weights["W1"] = np.random.randn(
            layers[0], nx)*np.sqrt(2/nx)
        self.__weights["b1"] = np.zeros([layers[0], 1], dtype=float)
        if layers[0] <= 0 or type(layers[0]) is not int:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if layers[i] <= 0 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(
                i + 1)] = np.random.randn(layers[i],
                                          layers[i-1])*np.sqrt(2/layers[i-1])
            self.__weights["b{}".format(
                i + 1)] = np.zeros([layers[i], 1], dtype=float)

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights
