"""Relaxation of MTE problem to convex problem with larger parameter space."""

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


def solve_lp_convex(beta: float, k_bernstein: int = 11) -> dict[str, float]:
    """Solve linear program and convex relaxation."""
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

    # Unit ball constraint
    constraints.append(
        om.NonlinearConstraint(
            func=lambda x: np.sum((x - 0.5) ** 2),
            upper_bound=0.5,
        ),
    )

    params = res.lower_optres.x

    np.testing.assert_array_almost_equal(a_eq @ params, b_eq)

    res_convex = om.minimize(
        fun=objective,
        params=params,
        algorithm="ipopt",
        constraints=constraints,
    )

    return {"lp": res.lower_bound, "convex": res_convex.fun}


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    num_points: int


args = {
    "task1": _Arguments(
        path_to_plot=BLD / "figures" / "relaxation" / "relaxation_mte.html",
        num_points=250,
    ),
}


for id_, kwargs in args.items():

    @pytask.mark.relax()
    @task(name=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_relaxation_mte(
        path_to_plot: Annotated[Path, Product],
        num_points: int,
    ) -> None:
        """Task for solving original and relaxed convex problem."""
        beta_grid = np.linspace(0.6, 1, num_points)

        results = [solve_lp_convex(beta) for beta in beta_grid]

        data = pd.DataFrame(results, index=beta_grid)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["lp"],
                mode="lines",
                name="LP Solution",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["convex"],
                mode="lines",
                name="Convex Program Solution",
            ),
        )

        fig.write_html(path_to_plot)
