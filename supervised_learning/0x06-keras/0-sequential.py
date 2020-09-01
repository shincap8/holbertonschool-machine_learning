#!/usr/bin/env python3
"""Function that builds a neural network with the Keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    model = K.Sequential()
    l2 = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=l2, input_dim=nx))
    for i in range(len(layers)):
        if i > 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=l2))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
