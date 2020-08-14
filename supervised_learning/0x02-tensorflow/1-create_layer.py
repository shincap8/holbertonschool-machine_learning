#!/usr/bin/env python3
"""Function that returns the tensor output of the layer"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that returns the tensor output of the layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=W)
    return model(prev)
