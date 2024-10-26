"""Functions used for relaxation of problem."""

import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import optimagic as om  # type: ignore[import-untyped]
from jax import grad
from pyvmte.classes import Estimand  # type: ignore[import-untyped]
from pyvmte.config import (  # type: ignore[import-untyped]
    IV_SM,
    SETUP_SM_SHARP,
)
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
)
from scipy.optimize import minimize as scipy_minimize  # type: ignore[import-untyped]

from thesis.pyvmte.pyvmte_sims_config import Y0_AT, Y0_NT, Y1_AT, Y1_NT
from thesis.utilities import make_mtr_binary_iv


# Build matrices from scratch, then optimize.
# For the non-linear one, add the unit-ball constraint
def _linear(params, slope):
    return np.inner(params, slope)


pscore_lo = IV_SM.pscores[0]
pscore_hi = IV_SM.pscores[1]

u_hi_extra = 0.2

_make_m0 = partial(
    make_mtr_binary_iv,
    yd_at=Y0_AT,
    yd_nt=Y0_NT,
    pscore_lo=pscore_lo,
    pscore_hi=pscore_hi,
)

_make_m1 = partial(
    make_mtr_binary_iv,
    yd_at=Y1_AT,
    yd_nt=Y1_NT,
    pscore_lo=pscore_lo,
    pscore_hi=pscore_hi,
)


def generate_sphere_constraint(num_dims: int, k: int) -> om.NonlinearConstraint:
    """Generate a sphere constraint sum((x_i-c)**k) <= u tangent to unit box."""
    upper_bound = num_dims * (0.5) ** k

    def func(x):
        return np.sum((x - 0.5) ** k)

    return om.NonlinearConstraint(
        func=func,
        upper_bound=upper_bound,
        derivative=grad(func),
    )


def generate_sphere_constraint_scipy(
    num_dims: int,
    k: int,
) -> dict[str, str | Callable]:
    """Generate a sphere constraint sum((x_i-c)**k) <= u tangent to unit box."""
    upper_bound = num_dims * (0.5) ** k

    def func(x):
        return np.sum((x - 0.5) ** k)

    # return NonlinearConstraint(

    return {
        "type": "ineq",
        "fun": lambda x: upper_bound - func(x),  # scipy uses <= 0 by default
        "jac": grad(func),
        "hess": grad(grad(func)),
    }


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
        for dim in range(num_dims * 2)
    ]


def _create_full_matrix_with_diagonal(num_dims, val, diag_val):
    matrix = np.full((num_dims, num_dims), val)

    np.fill_diagonal(matrix, diag_val)

    return matrix


def solve_lp_convex(
    beta: float,
    algorithm: str,
    k_approximation: int,
    return_optimizer: bool = False,  # noqa: FBT001, FBT002
    k_bernstein: int = 11,
    scipy: bool = False,  # noqa: FBT001, FBT002
) -> dict[str, float] | Callable:
    """Solve linear program and convex relaxation."""
    num_dims = (k_bernstein + 1) * 2

    y0_c = 0.5 - beta / 2
    y1_c = 0.5 + beta / 2

    m0 = _make_m0(yd_c=y0_c)

    m1 = _make_m1(yd_c=y1_c)

    u_partition = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi_extra, 1])

    kwargs = {
        "target": Estimand(
            esttype="late",
            u_lo=pscore_lo,
            u_hi=pscore_hi,
            u_hi_extra=u_hi_extra,
        ),
        "identified_estimands": SETUP_SM_SHARP.identified_estimands,
        "basis_funcs": generate_bernstein_basis_funcs(k=k_bernstein),
        "instrument": IV_SM,
        "u_partition": u_partition,
        "m0_dgp": m0,
        "m1_dgp": m1,
    }

    res = identification(**kwargs)

    if res.success[0] is False:
        msg = "Identification failed. Skipping relaxation and returnin nan."
        warnings.warn(msg, stacklevel=2)
        return {"lp": np.nan, "convex": np.nan}

    c = res.lp_inputs["c"]
    a_eq = res.lp_inputs["a_eq"]
    b_eq = res.lp_inputs["b_eq"]

    objective = partial(_linear, slope=c)

    num_rows, _ = a_eq.shape

    # Equality constraints from MTR model
    constraints = [
        om.LinearConstraint(
            value=b_eq[i],
            weights=a_eq[i, :],
        )
        for i in range(num_rows)
    ]

    eq_constraints_scipy = [
        {
            "type": "eq",
            "fun": lambda x, i=i: a_eq[i] @ x - b_eq[i],
            "jac": lambda x, i=i: a_eq[i],  # noqa: ARG005
        }
        for i in range(num_rows)
    ]

    ineq_constraints_scipy = [
        generate_sphere_constraint_scipy(num_dims=num_dims, k=k_approximation),
    ]

    constraints.append(generate_sphere_constraint(num_dims=num_dims, k=k_approximation))

    params = res.lower_optres.x

    np.testing.assert_array_almost_equal(a_eq @ params, b_eq)

    if return_optimizer is True and scipy is True:
        return partial(
            scipy_minimize,
            fun=objective,
            x0=params,
            jac=lambda x: c,  # noqa: ARG005
            hess=lambda x: np.zeros((num_dims, num_dims)),  # noqa: ARG005
            constraints=eq_constraints_scipy + ineq_constraints_scipy,
        )

    if return_optimizer is True:
        return partial(
            om.minimize,
            fun=objective,
            params=params,
            constraints=constraints,
            jac=lambda x: c,  # noqa: ARG005
        )

    res_convex = om.minimize(
        fun=objective,
        params=params,
        algorithm=algorithm,
        constraints=constraints,
        jac=lambda x: c,  # noqa: ARG005
    )

    return {"lp": res.lower_bound, "convex": res_convex.fun}
