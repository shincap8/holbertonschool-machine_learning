#!/usr/bin/env python3
"""Function that builds a modified version of
the LeNet-5 architecture using tensorflow"""

import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of
    the LeNet-5 architecture using tensorflow"""
    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(X)
    maxpool1 = K.layers.MaxPool2D((2, 2), (2, 2))(conv1)
    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                            kernel_initializer='he_normal',
                            activation='relu')(maxpool1)
    maxpool2 = K.layers.MaxPool2D((2, 2), (2, 2))(conv2)
    flatten = K.layers.Flatten()(maxpool2)
    dense1 = K.layers.Dense(units=120, kernel_initializer='he_normal',
                            activation='relu')(flatten)
    dense2 = K.layers.Dense(units=84, kernel_initializer='he_normal',
                            activation='relu')(dense1)
    dense3 = K.layers.Dense(units=10, kernel_initializer='he_normal',
                            activation='softmax')(dense2)
    optimizer = K.optimizers.Adam()
    model = K.Model(inputs=X, outputs=dense3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model
