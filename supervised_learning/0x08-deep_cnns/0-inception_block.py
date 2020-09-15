#!/usr/bin/env python3
"""Function that builds an inception block as
described in Going Deeper with Convolutions (2014)"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block as
    described in Going Deeper with Convolutions (2014)"""
    conv_1 = K.layers.Conv2D(filters[0], kernel_size=(1, 1), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(A_prev)
    conv_2 = K.layers.Conv2D(filters[1], kernel_size=(1, 1), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(A_prev)
    conv_2 = K.layers.Conv2D(filters[2], kernel_size=(3, 3), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(conv_2)
    conv_3 = K.layers.Conv2D(filters[3], kernel_size=(1, 1), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(A_prev)
    conv_3 = K.layers.Conv2D(filters[4], kernel_size=(5, 5), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(conv_3)
    maxpool1 = K.layers.MaxPool2D((3, 3), (1, 1), padding='same')(A_prev)
    conv_4 = K.layers.Conv2D(filters[5], kernel_size=(1, 1), padding='same',
                               kernel_initializer='he_normal',
                               activation='relu')(maxpool1)
    mid_1 = K.layers.concatenate([conv_1, conv_2, conv_3, conv_4], axis = 3)
    return mid_1
