#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient descent"""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, shuffle=shuffle, verbose=verbose)
    return history
