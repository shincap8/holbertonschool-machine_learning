#!/usr/bin/env python3
""" from dict to dataframe """
import pandas as pd


Dict = {"First": [0.0, 0.5, 1.0, 1.5],
        "Second": ['one', "two", "three", "four"]}
df = pd.DataFrame(Dict, index=list("ABCD"))
