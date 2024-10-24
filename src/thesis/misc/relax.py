"""Functions used for relaxation of problem."""

import numpy as np
import optimagic as om  # type: ignore[import-untyped]


def generate_poly_constraints(num_dims: int) -> list[om.NonlinearConstraint]:
    """Generate 4th-order polynomial that relax the (scaled and centered) unit cube.

    For example, with num_dims = 2 the constraints are:
        (x-0.5)**4 + (y-c)**4 <= 0.5
        (x-c)**4 + (y-0.5)**4 <= 0.5
        (x-0.5)**4 + (y-(1-c))**4 <= 0.5
        (x-(1-c))**4 + (y-0.5)**4 <= 0.5
    where `c` is a constant chosen such that the unit cube is contained within
    the intersection.

    """
    center = 0.5

    k = 4

    ub = num_dims * center**2

    # Constant such that the (scaled and centered) unit cube is contained
    c = (ub - center**k * (num_dims - 1)) ** (1 / k)

    # Matrix where each row centers x_1, ..., x_n in a given constraint equation
    centering_matrix = np.concatenate(
        (
            _create_full_matrix_with_diagonal(num_dims, center, c),
            _create_full_matrix_with_diagonal(num_dims, center, 1 - c),
        ),
    )

    return [
        om.NonlinearConstraint(
            func=lambda x, dim=dim: np.sum((x - centering_matrix[dim]) ** k),
            upper_bound=ub,
        )
        for dim in range(num_dims)
    ]


def _create_full_matrix_with_diagonal(num_dims, val, diag_val):
    matrix = np.full((num_dims, num_dims), val)

    np.fill_diagonal(matrix, diag_val)

    return matrix
