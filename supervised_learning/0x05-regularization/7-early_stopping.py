#!/usr/bin/env python3
"""Function that creates a layer of a neural network using dropout"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that creates a layer of a neural network using dropout"""
    stop = False
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        stop = True
    return stop, count
