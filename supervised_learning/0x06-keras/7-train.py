#!/usr/bin/env python3
"""Function that trains a model using mini-batch gradient descent
also analyze validaiton data, learning decay and early stopping"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """return history"""
    callback = []
    if validation_data is not None and early_stopping:
        callback = [K.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=patience)]
    if validation_data is not None and learning_rate_decay:
        def lr_scheduler(epoch):
            return (alpha / (1 + (decay_rate * epoch)))
        learning = K.callbacks.LearningRateScheduler(schedule=lr_scheduler,
                                                     verbose=1)
        callback.append(learning)
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
