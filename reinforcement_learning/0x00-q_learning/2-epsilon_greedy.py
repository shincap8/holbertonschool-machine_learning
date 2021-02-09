#!/usr/bin/env python3
"""Function that uses epsilon-greedy to determine the next action"""

import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return(action)
