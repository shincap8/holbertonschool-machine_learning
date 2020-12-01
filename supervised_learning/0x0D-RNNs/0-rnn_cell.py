#!/usr/bin/env python3
"""RNN cell class"""

import numpy as np


def softmax(x):
    """softmax function"""
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class RNNCell:
    """RNN cell class"""
    def __init__(self, i, h, o):
        """Constructor"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Method that performs forward propagation for one time step"""
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Wh) + self.bh)
        y = softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, y)
