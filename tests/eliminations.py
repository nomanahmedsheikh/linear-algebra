import numpy as np
from ops.reductions import elimination


def input_matrices():
    matrices = list()
    matrices.append(np.array([
        [1, 2, 3],
        [2, 2, 4],
        [3, 4, 2]
    ]))
    return matrices


def expected_outputs():
    matrices = list()
    matrices.append(np.array([
        [+1, +2, +3],
        [+0, -2, -2],
        [+0, +0, -5]
    ]))
    return matrices


def test_eliminations():
    inputs = input_matrices()
    outputs = list(map(lambda x: elimination(x), inputs))
    expected = expected_outputs()
    for output, actual in zip(outputs, expected):
        assert (output == actual).all()
