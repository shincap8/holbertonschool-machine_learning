#!/usr/bin/env python3
"""
Display a game played by the agent trained before
"""

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import keras as K
create_model = __import__('train').create_model
AtariProcessor = __import__('train').AtariProcessor

if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    window = 4  # number of screenshots
    model = create_model(num_actions, window)
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model,
                   nb_actions=num_actions,
                   test_policy=policy,
                   processor=processor,
                   memory=memory)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)
