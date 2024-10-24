"""Test relaxation functions."""

import numpy as np
from thesis.config import RNG
from thesis.misc.relax import generate_poly_constraints

c = (0.5 - (0.5) ** 4) ** (1 / 4)


def _fun_1(x):
    return (x[0] - c) ** 4 + (x[1] - 0.5) ** 4


def _fun_2(x):
    return (x[0] - 0.5) ** 4 + (x[1] - c) ** 4


def _fun_3(x):
    return (x[0] - (1 - c)) ** 4 + (x[1] - 0.5) ** 4


def _fun_4(x):
    return (x[0] - 0.5) ** 4 + (x[1] - (1 - c)) ** 4


def test_generate_poly_constraints() -> None:
    """Test function generating polynomial constraints."""

    num_dims = 2

    constraints = generate_poly_constraints(num_dims)

    assert len(constraints) == num_dims * 2

    actual_funcs = [constraint.func for constraint in constraints]

    expected_funcs = [_fun_1, _fun_2, _fun_3, _fun_4]

    for actual, expected in zip(actual_funcs, expected_funcs, strict=True):
        for _ in range(1000):
            x = RNG.uniform(size=num_dims, low=-0.2, high=1.2)
            np.testing.assert_array_almost_equal(actual(x), expected(x))
