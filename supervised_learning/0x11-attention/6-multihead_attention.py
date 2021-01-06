#!/usr/bin/env python3
"""MultiHeadAttention class to perform multi head attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class"""
    def __init__(self, dm, h):
        """Class constructor"""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape
        is (batch_size, num_heads, seq_len, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Call Method"""
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return (output, weights)
