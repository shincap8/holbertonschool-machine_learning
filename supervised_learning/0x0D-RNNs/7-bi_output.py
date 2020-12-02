#!/usr/bin/env python3
"""Bidirectional cell class"""

import numpy as np


def softmax(x):
    """softmax function"""
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class BidirectionalCell:
    """Bidirectional cell class"""
    def __init__(self, i, h, o):
        """Constructor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Method that performs forward propagation for one time step"""
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Method  that calculates the hidden state
        in the backward direction for one time step"""
        h_prev = np.tanh(np.dot(np.hstack((h_next, x_t)), self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """Method that calculates all outputs for the RNN"""
        T = H.shape[0]
        Y = []
        for t in range(T):
            Y.append(softmax(np.dot(H[t], self.Wy) + self.by))
        return np.array(Y)
