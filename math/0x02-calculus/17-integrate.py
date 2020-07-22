#!/usr/bin/env python3
"""function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if type(poly) is list and type(C) is int:
        ans = [C]
        for i in range(len(poly)):
            if type(i) != int:
                return None
            if (poly[i] / (i + 1)) % 1 == 0:
                ans.append(int(poly[i] / (i + 1)))
            else:
                ans.append(poly[i] / (i + 1))
        return ans
    return None
