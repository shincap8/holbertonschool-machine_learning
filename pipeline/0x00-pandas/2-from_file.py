#!/usr/bin/env python3
"""loads data from a file as a pd.DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """ doc """
    return pd.read_csv(filename, sep=delimiter)
