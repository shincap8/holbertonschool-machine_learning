#!/usr/bin/env python3
"""Function that calculates the Shannon entropy
and P affinities relative to a data point"""

import numpy as np


P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Function that calculates the Shannon entropy
    and P affinities relative to a data point"""
    D, P, betas, H = P_init(X, perplexity)
    for i in range(X.shape[0]):
        Di = np.append(D[i, :i], D[i, i+1:])
        Hi, Pi = HP(Di, betas[i])
        bmin = None
        bmax = None
        Hdiff = Hi - H
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                bmin = betas[i].copy()
                if bmax is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + bmax) / 2
            else:
                bmax = betas[i].copy()
                if bmin is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + bmin) / 2
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.shape[0]]))] = Pi
    return (P.T + P) / (2 * X.shape[0])
