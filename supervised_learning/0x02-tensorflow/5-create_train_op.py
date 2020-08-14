#!/usr/bin/env python3
"""Function that creates the training operation for the network"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return (optimizer.minimize(loss))
