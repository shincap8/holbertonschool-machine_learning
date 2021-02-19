#!/usr/bin/env python3
"""python script to train an agent that can play Atari’s Breakout"""

import numpy as np
import gym

from PIL import Image

from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input
from keras.optimizers import Adam
import keras as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class AtariProcessor(Processor):
    """Atari Processor class"""

    def process_observation(self, observation):
        """Process observation method
        gray scale and resizing"""
        assert observation.ndim == 3
        # (height, width, channel)
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """Process state batch method to rescale"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Process reward method reward between 1 and -1"""
        return np.clip(reward, -1., 1.)


def create_model(n_actions, window):
    """Method to Create the model for the Q-learning"""
    inputs = Input(shape=(window, 84, 84))
    model = Permute((2, 3, 1))(inputs)
    model = Conv2D(32, 8, strides=4, activation="relu",
                   data_format="channels_last")(model)
    model = Conv2D(64, 4, strides=2, activation="relu",
                   data_format="channels_last")(model)
    model = Conv2D(64, 3, strides=1, activation="relu",
                   data_format="channels_last")(model)
    model = Flatten()(model)
    model = Dense(512, activation="relu")(model)
    output = Dense(n_actions, activation="linear")(model)
    return K.Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    # Get the environment and extract the number of
    # actions available in the Breakout problem
    env = gym.make("Breakout-v0")
    env.seed(1)
    env.reset()
    nb_actions = env.action_space.n
    model = create_model(nb_actions, 4)
    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   memory=memory, processor=processor, nb_steps_warmup=1000,
                   gamma=.99, target_model_update=100,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=175000, log_interval=10000,
            visualize=False, verbose=2)
    model.save_weights('policy.h5', overwrite=True)
