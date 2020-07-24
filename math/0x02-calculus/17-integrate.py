#!/usr/bin/env python3
"""function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if poly is None or poly == [] or type(poly) != list:
        return None
    if type(C) is int or type(C) is float:
        if poly == [0]:
            return C
        ans = [C]
        for i in range(len(poly)):
            if type(poly[i]) != int and type(poly[i]) != float:
                return None
            if (poly[i] / (i + 1)) % 1 == 0:
                ans.append(int(poly[i] / (i + 1)))
            else:
                ans.append(poly[i] / (i + 1))
        for i in range(len(ans) - 1, 0, -1):
            if ans[i] == 0:
                ans.pop()
        return ans
    return None
