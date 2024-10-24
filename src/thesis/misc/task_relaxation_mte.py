"""Relaxation of MTE problem to convex problem with larger parameter space."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import optimagic as om  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task
from pyvmte.classes import Estimand  # type: ignore[import-untyped]
from pyvmte.config import (  # type: ignore[import-untyped]
    IV_SM,
    SETUP_SM_SHARP,
)
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
)

from thesis.config import BLD
from thesis.misc.relax import generate_poly_constraints
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


def solve_lp_convex(
    beta: float,
    algorithm: str,
    constraint: str,
    k_bernstein: int = 11,
    return_optimizer: bool = False,  # noqa: FBT001, FBT002
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

    if constraint == "unit_ball":
        # Unit ball constraint
        constraints.append(
            om.NonlinearConstraint(
                func=lambda x: np.sum((x - 0.5) ** 2),
                upper_bound=(k_bernstein + 1) * 2 * 0.25,
            ),
        )

    if constraint == "polynomial_degree4":
        constraints.extend(generate_poly_constraints(num_dims=num_dims))

    params = res.upper_optres.x

    np.testing.assert_array_almost_equal(a_eq @ params, b_eq)

    if return_optimizer:
        return partial(
            om.maximize,
            fun=objective,
            params=params,
            algorithm=algorithm,
            constraints=constraints,
        )

    res_convex = om.maximize(
        fun=objective,
        params=params,
        algorithm=algorithm,
        constraints=constraints,
    )

    return {"lp": res.upper_bound, "convex": res_convex.fun}


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    num_points: int
    k_bernstein: int


args = {
    f"k_bernstein_{k}": _Arguments(
        path_to_plot=BLD
        / "figures"
        / "relaxation"
        / f"relaxation_mte_k_bernstein_{k}.html",
        num_points=1000,
        k_bernstein=k,
    )
    for k in [2, 5, 11]
}


for id_, kwargs in args.items():

    @pytask.mark.relax()
    @task(name=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_relaxation_mte(
        path_to_plot: Annotated[Path, Product],
        num_points: int,
        k_bernstein: int,
    ) -> None:
        """Task for solving original and relaxed convex problem."""
        beta_grid = np.linspace(-1, 1, num_points)

        results_unit_ball = [
            solve_lp_convex(
                beta=beta,
                k_bernstein=k_bernstein,
                algorithm="scipy_slsqp",
                constraint="unit_ball",
            )
            for beta in beta_grid
        ]

        result_polynomial_degree4 = [
            solve_lp_convex(
                beta=beta,
                k_bernstein=k_bernstein,
                algorithm="scipy_slsqp",
                constraint="polynomial_degree4",
            )
            for beta in beta_grid
        ]

        data_unit_ball = pd.DataFrame(results_unit_ball, index=beta_grid)
        data_polynomial_degree4 = pd.DataFrame(
            result_polynomial_degree4,
            index=beta_grid,
        )
        data_unit_ball["constraint"] = "unit_ball"
        data_polynomial_degree4["constraint"] = "polynomial_degree4"

        data = pd.concat([data_unit_ball, data_polynomial_degree4])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[data["constraint"] == "unit_ball"].index,
                y=data[data["constraint"] == "unit_ball"]["lp"],
                mode="lines",
                name="LP Solution",
            ),
        )

        for constraint in ["unit_ball", "polynomial_degree4"]:
            fig.add_trace(
                go.Scatter(
                    x=data[data["constraint"] == constraint].index,
                    y=data[data["constraint"] == constraint]["convex"],
                    mode="lines",
                    name=f"Convex Program Solution ({constraint})",
                ),
            )

        fig.write_html(path_to_plot)
