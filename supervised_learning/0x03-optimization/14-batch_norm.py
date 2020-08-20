#!/usr/bin/env python3
"""Function that creates a batch normalization
layer for a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization
    layer for a neural network in tensorflow"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, name="layer", kernel_initializer=W)
    X = model(prev)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    mean, variance = tf.nn.moments(X, [0])
    norm = tf.nn.batch_normalization(X, mean, variance, offset=beta,
                                     scale=gamma, variance_epsilon=1e-8)
    return activation(norm)
