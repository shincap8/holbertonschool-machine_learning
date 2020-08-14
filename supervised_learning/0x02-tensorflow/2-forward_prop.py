#!/usr/bin/env python3
"""Function that creates the forward
propagation graph for the neural network"""


import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward
    propagation graph for the neural network"""
    A = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A
