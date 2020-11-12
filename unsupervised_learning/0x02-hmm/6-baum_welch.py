#!/usr/bin/env python3
"""Function that performs the Baum-Welch algorithm for a hidden markov model"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward
    algorithm for a hidden markov model"""
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros([N, T])
    B[:, T - 1] = np.ones((N))
    for i in range(T - 2, -1, -1):
        for j in range(N):
            B[j, i] = (B[:, i + 1] *
                       Emission[:, Observation[i + 1]]).dot(Transition[j, :])
    return (B)


def forward(Observation, Emission, Transition, Initial):
    """Function that performs the forward
    algorithm for a hidden markov model"""
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros([N, T])
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            F[j, i] = F[:, i - 1].dot(Transition[:, j]) *\
                Emission[j, Observation[i]]
    return (F)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Function hat performs the Baum-Welch
    algorithm for a hidden markov model"""
    T = Observations.shape[0]
    M, N = Emission.shape
    for n in range(1, iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((M, M, T - 1))
        for i in range(T - 1):
            den = np.dot(np.dot(F[:, i].T, Transition) *
                         Emission[:, Observations[i + 1]].T, B[:, i + 1])
            for j in range(M):
                num = F[j, i] * Transition[j] *\
                    Emission[:, Observations[i + 1]].T * B[:, i + 1].T
                xi[j, :, i] = num / den
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denom = np.sum(gamma, axis=1)
        for i in range(N):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
        Emission = np.divide(Emission, denom.reshape((-1, 1)))
    return (Transition, Emission)
