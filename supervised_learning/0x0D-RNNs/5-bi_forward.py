#!/usr/bin/env python3
"""Bidirectional cell class"""

import numpy as np


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
