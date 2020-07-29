#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson:
    """Poisson"""
    def __init__(self, data=None, lambtha=1.):
        """class constructor"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = (sum(data)/len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'"""
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, int(k)+1):
            factorial = factorial * i
        return (((2.7182818285**-(self.lambtha))*(self.lambtha**int(k)))
                / factorial)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of 'successes'"""
        if k <= self.n:
            return 0
        prob = 0
        for i in range(int(k)+1):
            prob = prob + self.pmf(i)
        return (prob)
