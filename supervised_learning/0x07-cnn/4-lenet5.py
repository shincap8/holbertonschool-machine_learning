#!/usr/bin/env python3
"""Function that builds a modified version of
the LeNet-5 architecture using tensorflow"""

import tensorflow as tf


def lenet5(x, y):
    """Function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    conv1 = tf.layers.conv2d(x, 6, kernel_size=(5, 5), padding='same', kernel_initializer=init, activation=tf.nn.relu)
    maxpool1 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    maxpool1 = maxpool1(conv1)
    conv2 = tf.layers.conv2d(maxpool1, 16, kernel_size=(
        5, 5), padding='valid', kernel_initializer=init, activation=tf.nn.relu)
    maxpool2 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    maxpool2 = maxpool2(conv2)
    flatten = tf.layers.flatten(maxpool2)
    dense1 = tf.layers.Dense(units=120, kernel_initializer=init, activation=tf.nn.relu)
    dense1 = dense1(flatten)
    dense2 = tf.layers.Dense(units=84, kernel_initializer=init, activation=tf.nn.relu)
    dense2 = dense2(dense1)
    dense3 = tf.layers.Dense(units=10, kernel_initializer=init, activation=tf.nn.relu)
    dense3 = dense3(dense2)
    softmax = tf.nn.softmax(dense3)
    optimizer = tf.train.AdamOptimizer()
    loss = tf.losses.softmax_cross_entropy(y, softmax)
    acc = tf.metrics.accuracy(y, softmax)
    return (softmax, optimizer, loss, acc)
