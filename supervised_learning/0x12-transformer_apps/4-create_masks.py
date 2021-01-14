#!/usr/bin/env python3
"""Function that creates all masks for training/validation"""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Function that creates all masks for training/validation"""
    enc_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]
    dec_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]
    size = target.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    dec_target_p_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_p_mask = dec_target_p_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_p_mask, look_ahead_mask)
    return (enc_padding_mask, combined_mask, dec_padding_mask)
