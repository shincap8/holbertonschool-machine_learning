#!/usr/bin/env python3
"""train method"""

import gym
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training
    parameters:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True render the environment
        every 1000 episodes computed
        Return:
            all values of the score (sum of all
            rewards during one episode loop)
    """
    w = np.random.rand(4, 2)
    episode_rewards = []
    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
        while True:
            if show_result and (e % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, w)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state
            if done:
                break
        for i in range(len(grads)):
            w += alpha * grads[i] *\
                sum([r * gamma ** r for t, r in enumerate(rewards[i:])])
        episode_rewards.append(score)
        print("{}: {}".format(e, score), end="\r", flush=False)
    return episode_rewards
