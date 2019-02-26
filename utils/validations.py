def validate_matrix(A):
    assert len(A.shape) is 2


def validate_point_in_matrix(point, A):
    validate_matrix(A)
    (m, n) = A.shape
    (i, j) = point
    assert m > i >= 0
    assert n > j >= 0
