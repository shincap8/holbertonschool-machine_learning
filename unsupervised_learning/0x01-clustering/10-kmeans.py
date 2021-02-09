#!/usr/bin/env python3
"""Function that performs K-means on a dataset"""

import sklearn.cluster


def kmeans(X, k):
    """X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index
        of the cluster in C that each data point belongs to"""
    kmean = sklearn.cluster.KMeans(k)
    kmean.fit(X)
    return kmean.cluster_centers_, kmean.labels_
