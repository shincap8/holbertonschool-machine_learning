#!/usr/bin/env python3
"""Dataset class that loads and preps a dataset for machine translation"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class"""
    def __init__(self, batch_size, max_len):
        """Constructor"""
        self.MAX_LENGTH = max_len
        data_train, data_info = tfds.load('ted_hrlr_translate/pt_to_en',
                                          split='train', as_supervised=True,
                                          with_info=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(self.fil_len)
        data_train = data_train.cache()
        num_examples = data_info.splits['train'].num_examples
        data_train = data_train.shuffle(num_examples).padded_batch(batch_size)
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True,)
        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(self.fil_len).padded_batch(batch_size)
        self.data_valid = data_valid

    def tokenize_dataset(self, data):
        """Method that creates sub-word tokenizers for our dataset"""
        subword = tfds.features.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = subword((en.numpy() for pt, en in data),
                               target_vocab_size=2**15)
        tokenizer_pt = subword((pt.numpy() for pt, en in data),
                               target_vocab_size=2**15)
        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """Method that encodes a translation into tokens"""
        pt = [self.tokenizer_pt.vocab_size] +\
            self.tokenizer_pt.encode(pt.numpy()) +\
            [self.tokenizer_pt.vocab_size+1]
        en = [self.tokenizer_en.vocab_size] +\
            self.tokenizer_en.encode(en.numpy()) +\
            [self.tokenizer_en.vocab_size+1]
        return (pt, en)

    def tf_encode(self, pt, en):
        """Method that acts as a tensorflow
        wrapper for the encode instance method"""
        result_pt, result_en = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return (result_pt, result_en)

    def fil_len(self, x, y):
        """Method to filter"""
        max_length = self.MAX_LENGTH
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)
