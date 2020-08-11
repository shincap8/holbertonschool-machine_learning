#!/usr/bin/env python3
"""class DeepNeuralNetwork that defines a deep neural
network performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            Z = np.dot(self.__weights["W{}".format(i)],
                       self.__cache["A{}".format(
                           i - 1)]) + self.__weights["b{}".format(i)]
            sigmoid = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i)] = sigmoid
        return (sigmoid, self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return ((-1 / A.shape[1]) * np.sum(loss))

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.round(A).astype(int), cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        dz = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
            dw = (np.matmul(cache["A{}".format(i - 1)], dz.T) / Y.shape[1])
            dz = np.matmul(self.__weights["W{}".format(
                i)].T, dz) * (cache["A{}".format(
                    i - 1)] * (1 - cache["A{}".format(i - 1)]))
            self.__weights["W{}".format(i)] = self.__weights["W{}".format(
                i)] - (alpha * dw).T
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(
                i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        axis_cost = []
        axis_it = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, A)
                axis_cost.append(cost)
                axis_it.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(axis_it, axis_cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
        return (self.evaluate(X, Y))

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if type(filename) is not str:
            return None
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, mode="wb") as f:
            pickle.dump(self, f)

    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if os.path.isfile('./' + filename):
            with open(filename, mode="rb") as f:
                return pickle.load(f)
        return None
