#!/usr/bin/env python3
"""Policy and policy_gradient methods"""

import numpy as np


def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix"""
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    return (exp / np.sum(exp))


def policy_gradient(state, weight):
    """Function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix
    parameters:
        state: matrix representing the current
        observation of the environment
        weight: matrix of random weight
        Return:
            the action and the gradient (in this order)
        """
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax[action, :]
    dlog = dsoftmax / P[0, action]
    grad = state.T.dot(dlog[None, :])
    return action, grad
