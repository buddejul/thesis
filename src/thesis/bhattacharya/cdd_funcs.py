"""Functions for determining extreme points of Polyhedra using pycddlib."""

import cdd
import numpy as np

# Rational for construction: https://pycddlib.readthedocs.io/en/latest/cdd.html#classes
# H-representation uses
# 0 <= b_i + A_ix   for inequality constraints and
# 0 = b_i + A_ix    for equality constraints.
# The matrix is then [b, A] where A is the matrix of coefficients of the x_i variables.
# Hence in the usual notation where A_ix <= b_i, we have to negate A.
# Representation is then in the form [t V] which represents the same polyhedron P.


def mat_box_constraint(d: int) -> np.ndarray:
    """Matrix representation of box constraint complying with cddlib."""
    a = -np.vstack([np.eye(d), -np.eye(d)])
    b = np.vstack([np.ones((d, 1)), np.zeros((d, 1))])

    return np.hstack([b, a])


def mat_box_constraint_with_equality(
    d: int,
    d_eq: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Matrix representation of box constraint with eq constr complying with cddlib."""
    mat_box = mat_box_constraint(d)

    # Add equality constraints of the type Ax = b where A has dimension d_eq x d
    # and b has dimension d_eq x 1
    _a = rng.normal(loc=0, scale=1, size=(d_eq, d))
    _b = _a @ rng.uniform(0, 1, (d, 1))

    # Equality constraints have to be rewritten as two inequalities
    a_eq = np.vstack([_a, -_a])
    b_eq = np.vstack([_b, -_b])

    mat_eq = np.hstack([b_eq, a_eq])

    return np.vstack([mat_box, mat_eq])


def find_extreme_points(mat_ineq_cdd: np.ndarray) -> np.ndarray:
    """Find extreme points of polyhedron defined by inequality matrix mat_ineq_cdd."""
    mat = cdd.matrix_from_array(mat_ineq_cdd, rep_type=cdd.RepType.INEQUALITY)  # type: ignore[arg-type]
    poly = cdd.polyhedron_from_matrix(mat)
    ext = cdd.copy_generators(poly)
    return np.array(ext.array)


def find_extreme_points_box(d: int) -> np.ndarray:
    """Find extreme points of box constraint in d dimensions."""
    mat_for_cdd = mat_box_constraint(d)
    return find_extreme_points(mat_for_cdd)


def find_extreme_points_box_with_equality(
    d: int,
    d_eq: int,
    rng: np.random.Generator,
) -> tuple:
    """Find extreme points of d-dim box constraint with d_eq equality constraints."""
    mat_for_cdd = mat_box_constraint_with_equality(d, d_eq, rng)

    ep = find_extreme_points(mat_for_cdd)
    if len(ep) == 0:
        return find_extreme_points_box_with_equality(d, d_eq, rng)

    return find_extreme_points(mat_for_cdd), mat_for_cdd
