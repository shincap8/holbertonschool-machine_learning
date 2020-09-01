#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent
also analyze validaiton data"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient descent
    also analyze validaiton data"""
    return network.fit(
                       x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data)
