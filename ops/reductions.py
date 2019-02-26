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
    print('Subtracting {} times Row[{}] from Row[{}]'.format(factor, pivot_row, i))
    A[i, :] = A[i, :] - factor * A[pivot_row, :]
    print(A)
    return A


def elimination(A):
    step_count = 1
    assert len(A.shape) == 2
    (m, n) = A.shape
    pivot_row = 0
    for j in range(n):
        if all(A[pivot_row:, j] == 0):
            continue
        if A[pivot_row, j] == 0:
            raise RuntimeError("Row Exchange not implemented")
        for i in range(pivot_row + 1, m):
            display_step(step_count)
            A = eliminate(A, pivot=(pivot_row, j), point=(i, j))
            step_count += 1
        pivot_row += 1
    return A


if __name__ == '__main__':
    A = np.ones((3, 4))
    A[0, :] = [1, 2, 3, 1]
    A[1, :] = [2, 2, 4, 1]
    A[2, :] = [3, 4, 2, 1]
    elimination(A)
