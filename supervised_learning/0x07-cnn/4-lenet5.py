#!/usr/bin/env python3
"""Function that builds a modified version of
the LeNet-5 architecture using tensorflow"""

import tensorflow as tf


def lenet5(x, y):
    """Function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                             kernel_initializer=init, activation=tf.nn.relu)
    conv1 = conv1(x)
    maxpool1 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    maxpool1 = maxpool1(conv1)
    conv2 = tf.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                             kernel_initializer=init, activation=tf.nn.relu)
    conv2 = conv2(maxpool1)
    maxpool2 = tf.layers.MaxPooling2D((2, 2), (2, 2))
    maxpool2 = maxpool2(conv2)
    flatten = tf.layers.Flatten()(maxpool2)
    dense1 = tf.layers.Dense(units=120, kernel_initializer=init,
                             activation=tf.nn.relu)
    dense1 = dense1(flatten)
    dense2 = tf.layers.Dense(units=84, kernel_initializer=init,
                             activation=tf.nn.relu)
    dense2 = dense2(dense1)
    dense3 = tf.layers.Dense(units=10, kernel_initializer=init)
    dense3 = dense3(dense2)
    softmax = tf.nn.softmax(dense3)
    y_p = tf.equal(tf.argmax(dense3, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(y_p, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, dense3)
    optimizer = tf.train.AdamOptimizer()
    adam = optimizer.minimize(loss)
    return (softmax, adam, loss, acc)
