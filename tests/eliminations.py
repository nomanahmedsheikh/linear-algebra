import numpy as np
from ops.reductions import elimination

MATRIX_0 = np.array([
    [1, 2, 3],
    [2, 2, 4],
    [3, 4, 2]])

MATRIX_1 = np.array([
    [1, 2, 1],
    [3, 8, 1],
    [0, 4, 1]])

MATRIX_2 = np.array([
    [1, 3],
    [2, 7]])

MATRIX_3 = np.array([
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 2]])

MATRIX_4 = np.array([
    [1, 2, 3, 1],
    [2, 4, 6, 4],
    [3, 6, 2, 1]])

MATRIX_REDUCED_0 = np.array([
    [+1, +2, +3],
    [+0, -2, -2],
    [+0, +0, -5]])

MATRIX_REDUCED_1 = np.array([
    [+1, +2, +1],
    [+0, +2, -2],
    [+0, +0, +5]])

MATRIX_REDUCED_2 = np.array([
    [1, 3],
    [0, 1]])

MATRIX_REDUCED_3 = np.array([
    [+1, +2, +3],
    [+0, +0, -7],
    [+0, +0, +0]])

MATRIX_REDUCED_4 = np.array([
    [+1, +2, +3, +1],
    [+0, +0, -7, -2],
    [+0, +0, -7, +0]])


def input_matrices():
    matrices = [MATRIX_0, MATRIX_1, MATRIX_2, MATRIX_3, MATRIX_4]
    return matrices


def expected_outputs():
    matrices = [MATRIX_REDUCED_0, MATRIX_REDUCED_1, MATRIX_REDUCED_2, MATRIX_REDUCED_3, MATRIX_REDUCED_4]
    return matrices


def test_eliminations():
    inputs = input_matrices()
    outputs = list(map(lambda x: elimination(x), inputs))
    expected = expected_outputs()
    for output, actual in zip(outputs, expected):
        assert (output == actual).all()
