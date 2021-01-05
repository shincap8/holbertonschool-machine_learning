#!/usr/bin/env python3
"""RNNEncoder class"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder class"""
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Method to Initializes the hidden states for
        the RNN cell to a tensor of zeros"""
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """Call Method"""
        embedding = self.embedding(x)
        outputs, hidden = self.gru(embedding, initial_state=initial)
        return (outputs, hidden)
