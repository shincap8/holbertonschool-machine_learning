#!/usr/bin/env python3
"""Function that creates the training operation
for a neural network in tensorflow using the
RMSProp optimization algorithm"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function that creates the training operation
    for a neural network in tensorflow using the
    RMSProp optimization algorithm"""
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return (optimizer.minimize(loss))
