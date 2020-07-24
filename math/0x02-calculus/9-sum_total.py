#!/usr/bin/env python3
"""Function to calculate a summation"""


def summation_i_squared(n):
    """Function to calculate a summation"""
    if type(n) is int and n > 0:
        return int((n*(n+1)*((2*n)+1))/6)
    return None
