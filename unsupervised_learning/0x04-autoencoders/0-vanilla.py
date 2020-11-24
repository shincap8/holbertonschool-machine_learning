#!/usr/bin/env python3
"""Function that creates an autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates an autoencoder"""
    model_input = keras.layers.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(model_input)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_d = decoded
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    encoder = keras.models.Model(model_input, encoded)
    decoder = keras.models.Model(input_d, decoded)
    auto = keras.models.Model(model_input, decoder(encoder(model_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return (encoder, decoder, auto)
