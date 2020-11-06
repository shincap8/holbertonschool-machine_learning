#!/usr/bin/env python3
"""Function that performs agglomerative clustering on a dataset"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Function that performs agglomerative clustering on a dataset"""
    Z = scipy.cluster.hierarchy.linkage(X, method="ward")
    dendo = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.show()
    return clss
