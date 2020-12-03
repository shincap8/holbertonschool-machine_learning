#!/usr/bin/env python3
"""Bidirectional cell class"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Method that performs forward propagation for a bidirectional RNN"""
    T = X.shape[0]
    Hf = []
    Hb = []
    h_prev = h_0
    h_next = h_t
    for t in range(T):
        h_prev = bi_cell.forward(h_prev, X[t])
        h_next = bi_cell.backward(h_next, X[T - 1 - t])
        Hf.append(h_prev)
        Hb.append(h_next)
    Hb = [x for x in reversed(Hb)]
    Hf = np.array(Hf)
    Hb = np.array(Hb)
    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)
    return (H, Y)
