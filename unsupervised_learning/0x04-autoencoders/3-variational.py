#!/usr/bin/env python3
"""Function that creates a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder"""

    def sampling(args):
        """Sampling"""
        z_mean, z_log_sigma = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    def reconstruction_loss(true, pred):
        """Loss reconstruction"""
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        subs = keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma)
        kl_loss = 1 + z_log_sigma - subs
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    """Encoder"""
    inputs = keras.layers.Input(shape=(input_dims,))
    encoded = inputs
    for i in range(len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])
    """Decoder"""
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_d = decoded
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(input_d, decoded)
    """auto"""
    outputs = decoder(encoder(inputs))
    auto = keras.models.Model(inputs, outputs)
    auto.compile(optimizer='adam', loss=reconstruction_loss)
    return (encoder, decoder, auto)
