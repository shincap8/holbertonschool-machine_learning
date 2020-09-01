#!/usr/bin/env python3
"""Function  that sets up Adam optimization for a keras
model with categorical crossentropy loss and accuracy metrics"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Function  that sets up Adam optimization for a keras
    model with categorical crossentropy loss and accuracy metrics"""
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return None
