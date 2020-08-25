#!/usr/bin/env python3
"""Function that calculates the precision
for each class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """Function that calculates the precision
    for each class in a confusion matrix"""
    return (confusion.diagonal() / np.sum(confusion, axis=0))
