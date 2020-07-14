#!/usr/bin/env python3
def cat_arrays(arr1, arr2):
    concat = list(arr1)
    for i in arr2:
        concat.append(i)
    return concat
