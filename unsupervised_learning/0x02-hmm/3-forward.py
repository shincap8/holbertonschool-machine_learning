#!/usr/bin/env python3
"""Function that performs the forward algorithm for a hidden markov model"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Function that performs the forward
    algorithm for a hidden markov model"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return (None, None)
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return (None, None)
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return (None, None)
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return (None, None)
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != Transition.shape[1] or Transition.shape[0] != N:
        return (None, None)
    if N != Initial.shape[0] or Initial.shape[1] != 1:
        return (None, None)
    F = np.zeros([N, T])
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    ll = 1
    for i in range(1, T):
        ll_O = 0
        for j in range(N):
            F[j, i] = F[:, i - 1].dot(Transition[:, j]) *\
                Emission[j, Observation[i]]
            ll_O = ll_O + F[j, i]
        ll = ll_O
    return (ll, F)
