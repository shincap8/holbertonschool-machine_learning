#!/usr/bin/env python3
"""Function that builds the inception network as
described in Going Deeper with Convolutions (2014)"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network as
    described in Going Deeper with Convolutions (2014)"""
    inputs = K.Input(shape=(224, 224, 3))
    X = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                        strides=(2, 2),
                        kernel_initializer='he_normal',
                        activation='relu')(inputs)
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(X)
    X = K.layers.Conv2D(64, kernel_size=(1, 1), padding='same',
                        kernel_initializer='he_normal',
                        activation='relu')(X)
    X = K.layers.Conv2D(192, kernel_size=(3, 3), padding='same',
                        kernel_initializer='he_normal',
                        activation='relu')(X)
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(X)
    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(X)
    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    X = inception_block(X, [160, 112, 224, 24, 64, 64])
    X = inception_block(X, [128, 128, 256, 24, 64, 64])
    X = inception_block(X, [112, 144, 288, 32, 64, 64])
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(X)
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = inception_block(X, [384, 192, 384, 48, 128, 128])
    X = K.layers.AvgPool2D((7, 7), padding='same')(X)
    X = K.layers.Dropout(0.4)(X)
    X = K.layers.Dense(1000, activation='softmax')(X)
    model = K.Model(inputs, X)
    return model
