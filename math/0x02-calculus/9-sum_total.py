#!/usr/bin/env python3
"""Function to calculate a summation"""


def summation_i_squared(n):
    """Function to calculate a summation"""
    if type(n) is int or type(n) is float:
        if ((n*(n+1)*((2*n)+1))/6) % 1 == 0:
            return int((n*(n+1)*((2*n)+1))/6)
        return (n*(n+1)*((2*n)+1))/6
    return None
