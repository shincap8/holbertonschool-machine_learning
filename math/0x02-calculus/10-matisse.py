#!/usr/bin/env python3
"""Derivate the calculate"""


def poly_derivative(poly):
    """Derivate the calculate"""
    if len(poly) == 1:
        return [0]
    if len(poly) == 0:
        return None
    ans = []
    for i in range(1, len(poly)):
        if type(poly[i]) is not int:
            return None
        ans.append(poly[i]*i)
    return ans
