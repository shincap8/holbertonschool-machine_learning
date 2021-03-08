#!/usr/bin/env python3
""" from np array to dataframe """
import pandas as pd


def from_numpy(array):
    """ Creates a pd.DataFrame from a np.ndarray:
    - array is the np.ndarray from which you should create the pd.DataFrame
    - The columns of the pd.DataFrame should be labeled in alphabetical
      order and capitalized. There will not be more than 26 columns.
    - Returns: the newly created pd.DataFrame """
    abc = list("ABCDEFGHIJKLMNOPQRSTUVW")
    cols = array.shape[1]
    return pd.DataFrame(array, columns=abc[:cols])
