#!/usr/bin/env python3
"""class NeuralNetwork that defines a neural network
with one hidden layer performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """NeuralNetwork class"""

    def __init__(self, nx, nodes):
        """class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros([nodes, 1], dtype=float)
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        sigmoid1 = 1 / (1 + np.exp(-Z1))
        self.__A1 = sigmoid1
        Z2 = np.dot(self.__W2, sigmoid1) + self.__b2
        sigmoid2 = 1 / (1 + np.exp(-Z2))
        self.__A2 = sigmoid2

        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        loss = (Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))
        return ((-1 / A.shape[1]) * np.sum(loss))

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.round(A).astype(int), cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        dz2 = A2 - Y
        db2 = (np.sum(dz2, axis=1, keepdims=True) / X.shape[1])
        dw2 = (np.matmul(A1, dz2.T) / X.shape[1])
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        db1 = (np.sum(dz1, axis=1, keepdims=True) / X.shape[1])
        dw1 = (np.matmul(X, dz1.T) / X.shape[1])
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W2 = self.__W2 - (alpha * dw2).T
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W1 = self.__W1 - (alpha * dw1).T

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
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
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if i == 0 or i == iterations or i % step == 0:
                cost = self.cost(Y, self.__A2)
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
