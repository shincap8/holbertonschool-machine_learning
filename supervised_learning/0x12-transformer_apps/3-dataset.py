#!/usr/bin/env python3
"""Dataset class that loads and preps a dataset for machine translation"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class"""
    def __init__(self, batch_size, max_len):
        """ doc """
        examples, data_info = tfds.load('ted_hrlr_translate/pt_to_en',
                                        with_info=True,
                                        as_supervised=True)
        data_info = data_info
        data_train, data_valid = examples['train'], examples['validation']
        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(data_train)
        data_train = data_train.map(self.tf_encode)
        data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            """Filter max length method"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        train_dataset_size = data_info.splits['train'].num_examples
        data_train = data_train.shuffle(train_dataset_size)
        padded_shapes = ([None], [None])
        data_train = data_train.padded_batch(batch_size,
                                             padded_shapes=padded_shapes)
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
        data_valid = data_valid.filter(filter_max_length)
        padded_shapes = ([None], [None])
        self.data_valid = data_valid.padded_batch(batch_size,
                                                  padded_shapes=padded_shapes)

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
