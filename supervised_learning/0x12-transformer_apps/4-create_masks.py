#!/usr/bin/env python3
"""Function that creates all masks for training/validation"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_padding_mask(seq):
    """Function to create padding"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Function to mask future tokens"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """Function that creates all masks for training/validation"""
    enc_padding_mask = create_padding_mask(inputs)
    dec_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return (enc_padding_mask, combined_mask, dec_padding_mask)
