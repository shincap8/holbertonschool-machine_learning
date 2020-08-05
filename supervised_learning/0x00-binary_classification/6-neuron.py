#!/usr/bin/env python3
"""class Neuron that defines a single neuron
performing binary classification"""


import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
        """class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.dot(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-Z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return ((-1 / A.shape[1]) * np.sum(loss))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.round(A).astype(int), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        dz = A - Y
        db = (np.sum(dz) / X.shape[1])
        dw = (np.matmul(X, dz.T) / X.shape[1])
        self.__b = self.__b - (alpha * db)
        self.__W = self.__W - (alpha * dw).T

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return (self.evaluate(X, Y))
