#!/usr/bin/env python3
"""Function that builds a neural network with the Keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=l2)(inputs)
    for i in range(len(layers)):
        if i > 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=l2)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    return model
