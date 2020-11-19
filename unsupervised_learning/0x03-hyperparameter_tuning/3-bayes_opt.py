#!/usr/bin/env python3
"""Class BayesianOptimization Bayesian
optimization on a noiseless 1D Gaussian process"""

import numpy as np
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
        step = (bounds[1] - bounds[0]) / (ac_samples - 1)
        Xs = np.zeros([ac_samples, 1])
        Xs[0] = bounds[0]
        for i in range(1, ac_samples):
            Xs[i] = Xs[i - 1] + step
        self.X_s = Xs