#!/usr/bin/env python3
"""Function that calculates the sensitivity
for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity
    for each class in a confusion matrix"""
    return (confusion.diagonal() / np.sum(confusion, axis=1))
