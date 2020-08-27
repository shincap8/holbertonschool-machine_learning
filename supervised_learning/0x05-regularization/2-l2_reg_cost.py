#!/usr/bin/env python3
"""Function that calculates the cost of a
neural network with L2 regularization"""


import tensorflow as tf


def l2_reg_cost(cost):
    """Function that calculates the cost of a
    neural network with L2 regularization"""
    return cost + tf.losses.get_regularization_losses()
