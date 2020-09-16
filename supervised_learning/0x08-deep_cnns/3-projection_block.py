#!/usr/bin/env python3
"""Function that builds a projection block as described
in Deep Residual Learning for Image Recognition (2015)"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)"""
    X = K.layers.Conv2D(filters[0], kernel_size=(1, 1), padding='same',
                        kernel_initializer='he_normal', strides=(s, s))(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters[1], kernel_size=(3, 3), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters[2], kernel_size=(1, 1), padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    shortcut = K.layers.Conv2D(filters[2], kernel_size=(1, 1), padding='same',
                               kernel_initializer='he_normal',
                               strides=(s, s))(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)
    return X
