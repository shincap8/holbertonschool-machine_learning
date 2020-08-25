#!/usr/bin/env python3
"""Function that calculates the specificity
for each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """Function that calculates the specificity
    for each class in a confusion matrix"""
    FN = (np.sum(confusion, axis=1) - confusion.diagonal())
    FP = (np.sum(confusion, axis=0) - confusion.diagonal())
    TN = np.sum(confusion) - (FP + FN + confusion.diagonal())
    return (TN / (FP + TN))
