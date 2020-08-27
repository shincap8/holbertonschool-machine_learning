#!/usr/bin/env python3
"""Function that creates a layer of a neural network using dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of a neural network using dropout"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.layers.Dropout(keep_prob)
    model = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=W,
                            kernel_regularizer=l2)
    return model(prev)
