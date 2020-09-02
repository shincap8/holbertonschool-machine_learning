#!/usr/bin/env python3
"""Functions to load and save a model"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network"""
    results = network.evaluate(data, labels, verbose=verbose)
    return results
