#!/usr/bin/env python3
"""Class BayesianOptimization Bayesian
optimization on a noiseless 1D Gaussian process"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Class BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.minimize = minimize
        self.xsi = xsi
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """Method that calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize:
                musopt = np.min(self.gp.Y)
                imp = (musopt - mu - self.xsi).reshape(-1, 1)
            else:
                musopt = np.amax(self.gp.Y)
                imp = (mu - musopt - self.xsi).reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return (X_next, ei.reshape(-1))

    def optimize(self, iterations=100):
        """Method that calculates the next best sample location"""
        for i in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if (X_next == self.gp.X).any():
                self.gp.X = self.gp.X[:-1]
                break
            self.gp.update(X_next, Y_next)
        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        Y_opt = self.gp.Y[index]
        X_opt = self.gp.X[index]
        return (X_opt, Y_opt)
