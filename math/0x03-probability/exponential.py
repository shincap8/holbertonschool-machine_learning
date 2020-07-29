#!/usr/bin/env python3
"""class Exponential that represents an exponential distribution"""


class Exponential:
    """Exponential"""
    def __init__(self, data=None, lambtha=1.):
        """class constructor"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = (1/(sum(data)/len(data)))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        return (self.lambtha*(2.7182818285**-(self.lambtha * x)))

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0
        return (1 - (2.7182818285**-(self.lambtha * x)))
