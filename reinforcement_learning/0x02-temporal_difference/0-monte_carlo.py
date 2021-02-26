#!/usr/bin/env python3
"""Monte Carlo Method"""

import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Function that performs the Monte Carlo algorithm
    Parameters
    env: is the openAI environment instance
    V: is a numpy.ndarray of shape (s,) containing the value estimate
    policy: is a function that takes in a state
        and returns the next action to take
    episodes: is the total number of episodes to train over
    max_steps: is the maximum number of steps per episode
    alpha: is the learning rate
    gamma: is the discount rate

    Returns: V
    V: the updated value estimate"""
    space_n = env.observation_space.n
    discounts = [gamma**i for i in range(max_steps)]
    for episode in range(episodes):
        state = env.reset()
        episode = [[], []]
        for i in range(max_steps):
            action = policy(state)
            new_s, reward, _, _ = env.step(action)
            episode[0].append(state)
            if env.desc.reshape(space_n)[new_s] == b'G':
                episode[1].append(1)
                break
            if env.desc.reshape(space_n)[new_s] == b'H':
                episode[1].append(-1)
                break
            episode[1].append(reward)
            state = new_s
        for i in range(len(episode[0])):
            Gt = sum(np.array(episode[1][i:]) *
                     np.array(discounts[:len(episode[1][i:])]))
            V[episode[0][i]] = V[episode[0][i]] +\
                alpha * (Gt - V[episode[0][i]])
    return V
