#!/usr/bin/env python3
"""Function that calculates the F1 score of a confusion matrix"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function that calculates the
    F1 score of a confusion matrix"""
    den = (sensitivity(confusion) ** -1) + (precision(confusion) ** -1)
    return (2 / den)
