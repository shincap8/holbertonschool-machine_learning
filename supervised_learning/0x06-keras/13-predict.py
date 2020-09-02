#!/usr/bin/env python3
"""Functions that makes a prediction using a neural network"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network"""
    results = network.predict(data, verbose=verbose)
    return results
