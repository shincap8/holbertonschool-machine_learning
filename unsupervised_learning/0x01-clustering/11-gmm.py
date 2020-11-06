#!/usr/bin/env python3
"""Function that calculates a GMM from a dataset"""

import sklearn.mixture


def gmm(X, k):
    """Function that calculates a GMM from a dataset"""
    Gmm = sklearn.mixture.GaussianMixture(k)
    params = Gmm.fit(X)
    clss = Gmm.predict(X)
    return (params.weights_, params.means_,
            params.covariances_, clss, Gmm.bic(X))
