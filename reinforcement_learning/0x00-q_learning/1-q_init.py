#!/usr/bin/env python3
"""Function that initializes the Q-table"""

import gym
import numpy as np


def q_init(env):
    """Function that initializes the Q-table"""
    a_space_size = env.action_space.n
    s_space_size = env.observation_space.n
    q_table = np.zeros((s_space_size, a_space_size))
    return(q_table)
