#!/usr/bin/env python3
def matrix_transpose(matrix):
    matrixT = []
    start = True
    for row in matrix:
        for i in range(len(row)):
            if start == True:
                matrixT.append(list())
            matrixT[i].append(row[i])
        start = False
    return (matrixT)
