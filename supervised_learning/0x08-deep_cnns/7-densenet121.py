#!/usr/bin/env python3
"""Function that builds the DenseNet-121 architecture as
described in Densely Connected Convolutional Networks"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture as
    described in Densely Connected Convolutional Networks"""
    inputs = K.Input(shape=(224, 224, 3))
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                        kernel_initializer='he_normal', strides=(2, 2))(X)
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding="same")(X)
    X, filters = dense_block(X, 64, growth_rate, 6)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 12)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 24)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 16)
    X = K.layers.AvgPool2D((7, 7), padding='same')(X)
    X = K.layers.Dense(1000, activation='softmax')(X)
    model = K.Model(inputs, X)
    return model
