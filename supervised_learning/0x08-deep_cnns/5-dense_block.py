#!/usr/bin/env python3
"""Function that builds a dense block as described
in Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described
    in Densely Connected Convolutional Networks"""
    concatenate = inputs
    for i in range(layers):
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                            kernel_initializer='he_normal', strides=(2, 2))(inputs)
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                            kernel_initializer='he_normal', strides=(2, 2))(inputs)
        concatenate = K.layers.concatenate([concatenate, X], axis=3)
    return 
