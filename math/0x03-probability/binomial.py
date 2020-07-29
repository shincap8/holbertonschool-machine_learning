#!/usr/bin/env python3
"""class Binomial that represents a binomial distribution"""


class Binomial:
    """Binomial"""

    def __init__(self, data=None, n=1, p=0.5):
        """class constructor"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 and p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = (sum(data)/len(data))
            dif = []
            for i in data:
                dif.append((i - mean) ** 2)
            var = sum(dif) / len(data)
            p = 1 - (var / mean)
            oldn = mean / p
            self.n = int(round(mean / p))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'"""
        if k <= 0:
            return 0
        factor1, factor2, factor3 = 1, 1, 1
        for i in range(1, (self.n + 1)):
            factor1 = factor1 * i
        for i in range(1, (k + 1)):
            factor2 = factor2 * i
        for i in range(1, (self.n - k + 1)):
            factor3 = factor3 * i
        return ((factor1 / (factor2 * factor3)) * (self.p ** k) *
                ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of 'successes'"""
        if k < 0:
            return 0
        prob = 0
        for i in range(int(k)+1):
            prob = prob + self.pmf(i)
        return (prob)
