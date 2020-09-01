#!/usr/bin/env python3
"""Function that builds a neural network with the Keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    dense = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=l2)
    x = dense(inputs)
    for i in range(len(layers) - 1):
        if i > 0 and i < len(layers) - 1:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=l2)(x)
        x = K.layers.Dropout(1 - keep_prob)(x)
    outputs = K.layers.Dense(layers[len(layers) - 1],
                             activation=activations[len(layers) - 1],
                             kernel_regularizer=l2)(x)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model
