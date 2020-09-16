#!/usr/bin/env python3
"""Function that builds the ResNet-50 architecture asdescribed
in Deep Residual Learning for Image Recognition (2015)"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)"""
    inputs = K.Input(shape=(224, 224, 3))
    X = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                        kernel_initializer='he_normal', strides=(2, 2))(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPool2D((3, 3), (2, 2), padding="same")(X)
    X = projection_block(X, [64, 64, 256], 1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = K.layers.AvgPool2D((7, 7), padding='same')(X)
    X = K.layers.Dense(1000, activation='softmax')(X)
    model = K.Model(inputs, X)
    return model
