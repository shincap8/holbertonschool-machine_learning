#!/usr/bin/env python3
"""Function that calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set"""
    V = [0]
    for i in range(len(data)):
        V.append((beta * V[i]) + ((1 - beta) * data[i]))
    moving = []
    for i in range(1, len(V)):
        moving.append(V[i] / (1 - (beta ** i)))
    return moving
