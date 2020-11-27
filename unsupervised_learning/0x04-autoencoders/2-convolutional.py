#!/usr/bin/env python3
"""Function that creates a convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder"""
    model_input = keras.layers.Input(shape=input_dims)
    encoded = model_input
    for i in range(len(filters)):
        encoded = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    decoded = keras.layers.Input(shape=latent_dims)
    input_d = decoded
    padding = 'same'
    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            padding = 'valid'
        decoded = keras.layers.Conv2D(filters[i], (3, 3), activation='relu',
                                      padding=padding)(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(decoded)
    encoder = keras.models.Model(model_input, encoded)
    decoder = keras.models.Model(input_d, decoded)
    auto = keras.models.Model(model_input, decoder(encoder(model_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return (encoder, decoder, auto)
