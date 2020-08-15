#!/usr/bin/env python3
"""Function that calculates the accuracy of a prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    return tf.math.reduce_mean((y - y_pred)**2)
