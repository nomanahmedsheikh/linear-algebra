"""
All the reduction operations that could be performed on a Matrix
Most often, these operation would include:
    1. Converting a matrix into Upper Triangular Form without changing Row Space
    2. Converting a matrix into Row Reduced Echelon Form
    3. Finding Inverse using Augmented Matrix
    4. LD Decomposition
    5. LDU Decomposition
"""

import numpy as np
from utils.validations import *
from utils.stdout import *


def eliminate(A, pivot, point):
    validate_point_in_matrix(pivot, A)
    validate_point_in_matrix(point, A)
    print('Eliminating {} at position {}'.format(A[point[0], point[1]], point))
    # row_i = row_i - factor * row_pivot
    factor = A[point[0], point[1]] / A[pivot[0], pivot[1]]
    i = point[0]
    pivot_row = pivot[0]
    print('By Subtracting {} times Row[{}] from Row[{}]'.format(factor, pivot_row, i))
    A[i, :] = A[i, :] - factor * A[pivot_row, :]
    return A


def next_pivot_row(matrix, pivot_row, pivot_col):
    column = matrix[:, pivot_col]
    for row_index in range(pivot_row + 1, len(column)):
        if matrix[row_index, pivot_col] is not 0:
            return row_index
    # This column need to be skipped
    return pivot_row


def swap_rows(matrix, row1, row2):
    print('Permuting Row[{}] and Row[{}]'.format(row1, row2))
    garbage = np.array(matrix[row1, :])
    matrix[row1, :] = np.array(matrix[row2, :])
    matrix[row2, :] = garbage
    return matrix


def elimination(A):
    step_count = 1
    assert len(A.shape) == 2
    (m, n) = A.shape
    pivot_row = 0
    for j in range(n):
        if A[pivot_row, j] == 0:
            k = next_pivot_row(matrix=A, pivot_row=pivot_row, pivot_col=j)
            if k is not pivot_row:
                step_count = display_step(step_count)
                A = swap_rows(matrix=A, row1=pivot_row, row2=k)
                display_matrix(matrix=A)
        if all(A[pivot_row+1:, j] == 0):
            continue
        for i in range(pivot_row + 1, m):
            step_count = display_step(step_count)
            A = eliminate(A, pivot=(pivot_row, j), point=(i, j))
            display_matrix(matrix=A)
        pivot_row += 1
    return A
