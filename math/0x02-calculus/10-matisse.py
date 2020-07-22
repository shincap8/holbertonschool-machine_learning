#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if type(poly) is not list:
        return None
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
