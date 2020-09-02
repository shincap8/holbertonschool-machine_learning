#!/usr/bin/env python3
"""Functions to load and save a model"""

import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format"""
    network.to_json(filename)
    return None


def load_config(filename):
    """Function that  loads a model with a specific configuration"""
    network.from_json(filename)
    return None
