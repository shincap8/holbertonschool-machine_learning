#!/usr/bin/env python3
"""Function that calculates the accuracy of a prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    return tf.reduce_mean(tf.cast(tf.equal((y - y_pred), 0), 'float32'))
