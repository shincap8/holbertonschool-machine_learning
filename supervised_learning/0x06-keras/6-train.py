#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """return history"""
    if validation_data is not None and early_stopping:
        callback = [K.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=patience)]
    else:
        callback = NULL
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callback)
    return history
