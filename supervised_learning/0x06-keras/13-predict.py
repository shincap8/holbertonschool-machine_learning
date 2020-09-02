#!/usr/bin/env python3
"""Functions that makes a prediction using a neural network"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that makes a prediction using a neural network"""
    results = network.predict(data, labels, verbose=verbose)
    return results
