#!/usr/bin/env python3
"""class Normal that represents a normal distribution"""


class Normal:
    """Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor"""
        self.stddev = float(stddev)
        self.mean = float(mean)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = ((sum(data)/len(data)))
            datai = []
            for i in data:
                datai.append((i - self.mean) ** 2)
            self.stddev = (((1 / len(data)) * sum(datai)) ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return ((z * self.stddev) + self.mean)

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        ex = -((1/2) * (((x - self.mean) / self.stddev) ** 2))
        e = 2.7182818285
        pi = 3.1415926536
        return ((1 / (self.stddev * ((2 * pi) ** 0.5))) * (e ** ex))

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        e = 2.7182818285
        pi = 3.1415926536
        ex = (x - self.mean) / (self.stddev * (2 ** 0.5))
        er = (2 / (pi ** 0.5)) * (ex - ((ex ** 3) / 3) + ((ex ** 5) / 10) -
                                  ((ex ** 7) / 42) + ((ex ** 9) / 216))
        return ((1/2) * (1 + er))
