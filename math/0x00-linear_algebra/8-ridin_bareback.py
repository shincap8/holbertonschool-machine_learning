#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

def mat_mul(mat1, mat2):
    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None
    mul = []
    shapem = [matrix_shape(mat1)[0], matrix_shape(mat2)
              [1], matrix_shape(mat1)[1]]
    i = 0
    j = 0
    for i in range(shapem[0]):
        mul.append(list())
        for j in range(shapem[1]):
            x = 0
            for k in range(shapem[2]):
                x += mat1[i][k] * mat2[k][j]
            mul[i].append(x)
    return mul


