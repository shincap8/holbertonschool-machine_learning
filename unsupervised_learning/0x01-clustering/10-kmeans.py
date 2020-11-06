#!/usr/bin/env python3
"""Function that performs K-means on a dataset"""

import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset"""
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
