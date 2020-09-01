#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent
also analyze validaiton data and early stopping"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """return history"""
    callback = []
    if validation_data is not None and early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=patience))
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
