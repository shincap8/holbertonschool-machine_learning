#!/usr/bin/env python3
"""Functions to load and save a model"""

import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format"""
    model = network.to_json()
    with open(filename, "w") as f:
        f.write(model)
    return None


def load_config(filename):
    """Function that  loads a model with a specific configuration"""
    with open(filename, "r") as f:
        model = f.read()
    network = K.models.model_from_json(model)
    return None
