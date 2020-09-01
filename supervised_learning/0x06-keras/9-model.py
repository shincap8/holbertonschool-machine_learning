#!/usr/bin/env python3
"""Functions to load and save a model"""

import tensorflow.keras as K


def save_model(network, filename):
    """Function that saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """Function that loads an entire model"""
    model = keras.models.load_model(filename)
    return model
