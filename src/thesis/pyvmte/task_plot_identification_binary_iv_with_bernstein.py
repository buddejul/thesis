"""Plot solutions to binary IV model showing the solution along the bounds."""

# TODO(@buddejul): Should make sure, that using constant spline MTRs as the DGP does not
# introduce the kinks we observe for the Bernstein solution.
# Alternative approach: Could we get the lower/upper bound Bernstein MTR from the
# identification function and then take some kind of average?

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from pytask import Product, task
from pyvmte.classes import Estimand  # type: ignore[import-untyped]
from pyvmte.config import IV_SM, SETUP_SM_IDLATE  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
)

from thesis.config import BLD
from thesis.pyvmte.pyvmte_sims_config import Y0_AT, Y0_NT, Y1_AT, Y1_NT
from thesis.utilities import make_mtr_binary_iv

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]

target = Estimand(esttype="late", u_lo=0.4, u_hi=0.6, u_hi_extra=0.2)

identified_estimands = SETUP_SM_IDLATE.identified_estimands

instrument = IV_SM

u_partition = np.array([0, 0.4, 0.6, 0.8, 1])

id_kwargs = {
    "target": target,
    "identified_estimands": identified_estimands,
    "instrument": instrument,
    "u_partition": u_partition,
}

pscore_lo, pscore_hi = instrument.pscores[0], instrument.pscores[1]

id_func = partial(identification, **id_kwargs)


def _generate_color_by_x_number(k_bernstein):
    # Cycle through the palette to create the color dictionary
    # Cycle through red numbers using rgb
    return {i: f"rgb({255 - i * 255 // k_bernstein}, 0, 0)" for i in range(k_bernstein)}


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    k_bernstein: int
    bounds_to_plot: tuple[str] = ("lower",)
    num_grid_points: int = 2_500


ID_TO_KWARGS = {
    f"degree_{k}": _Arguments(
        path_to_plot=BLD
        / "figures"
        / "binary_iv"
        / f"identification_with_bernstein_{k}.png",
        k_bernstein=k,
    )
    for k in [2, 3, 11]
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.plot_for_paper
    def task_plot_identification_binary_iv_with_bernstein(  # noqa: PLR0915, C901, PLR0912
        path_to_plot: Annotated[Path, Product],
        k_bernstein: int,
        bounds_to_plot: tuple[str],
        num_grid_points: int,
        show_legend: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Plot bounds and bfunc coefs for binary IV model and Bernstein."""
        basis_funcs = generate_bernstein_basis_funcs(k_bernstein)

        b_late_grid = np.linspace(-1, 1, num=num_grid_points)

        # ------------------------------------------------------------------------------
        # Compute solutions
        # ------------------------------------------------------------------------------

        res_lower_bounds = np.zeros_like(b_late_grid)
        res_upper_bounds = np.zeros_like(b_late_grid)

        optres_lower: dict[str, OptimizeResult] = {}
        optres_upper: dict[str, OptimizeResult] = {}

        for i, beta_late in enumerate(b_late_grid):
            y1_c = beta_late / 2 + 0.5
            y0_c = -beta_late / 2 + 0.5

            mtr1 = make_mtr_binary_iv(
                yd_c=y1_c,
                yd_at=Y1_AT,
                yd_nt=Y1_NT,
                pscore_lo=pscore_lo,
                pscore_hi=pscore_hi,
            )
            mtr0 = make_mtr_binary_iv(
                yd_c=y0_c,
                yd_at=Y0_AT,
                yd_nt=Y0_NT,
                pscore_lo=pscore_lo,
                pscore_hi=pscore_hi,
            )

            _res = id_func(basis_funcs=basis_funcs, m1_dgp=mtr1, m0_dgp=mtr0)

            res_lower_bounds[i] = _res.lower_bound
            res_upper_bounds[i] = _res.upper_bound

            optres_lower[beta_late] = _res.lower_optres
            optres_upper[beta_late] = _res.upper_optres

        # ------------------------------------------------------------------------------
        # Collect Data
        # ------------------------------------------------------------------------------
        data = pd.DataFrame(
            {
                "beta_late": b_late_grid,
                "lower_bound": res_lower_bounds,
                "upper_bound": res_upper_bounds,
            },
        )
        for bound in ["lower", "upper"]:
            _optres = optres_lower if bound == "lower" else optres_upper
            _mat = np.array([_optres[beta].x for beta in b_late_grid])
            data = data.assign(
                **{f"x_{i}_{bound}": _mat[:, i] for i in range(_mat.shape[1])},
            )

        # Identify kinks in "lower_bound" by looking at changes in the first
        # derivative. First, approximate the derivative by taking differences

        data["d_lower_bound"] = np.gradient(data["lower_bound"], b_late_grid)
        data["d_lower_bound"]

        # Find the kinks by looking at changes in the first derivative larger than
        # some epsilon Compute differences in the first derivative
        data["dd_lower_bound"] = np.gradient(data["d_lower_bound"], b_late_grid)

        # Find the kinks
        epsilon = 0.1
        kinks = data.loc[
            (data["dd_lower_bound"].abs() > epsilon),
            ["beta_late", "lower_bound", "d_lower_bound", "dd_lower_bound"],
        ]

        kinks_beta = kinks["beta_late"].to_numpy()

        # --------------------------------------------------------------------------
        # Figure
        # --------------------------------------------------------------------------

        # Example usage
        color_by_x_number = _generate_color_by_x_number(k_bernstein + 1)

        num_bounds_to_plot = len(bounds_to_plot)

        if num_bounds_to_plot == 1:
            rows = 3
            cols = 1
        else:
            rows = 3
            cols = 2

        if num_bounds_to_plot == 1:
            subplot_titles = (
                f"{bounds_to_plot[0].capitalize()} Bound",
                "Basis Coefficients for MTR d = 0",
                "Basis Coefficients for MTR d = 1",
            )
        else:
            subplot_titles = (  # type: ignore[assignment]
                "Bounds",
                "Basis Coefficients for MTR d = 0",
                "Basis Coefficients for MTR d = 1",
                "Basis Coefficients for MTR d = 0",
                "Basis Coefficients for MTR d = 1",
            )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
        )

        for bound in bounds_to_plot:
            fig.add_trace(
                go.Scatter(
                    x=data["beta_late"],
                    y=data[f"{bound}_bound"],
                    mode="lines",
                    name=f"{bound.capitalize()} Bound",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        for bound in bounds_to_plot:
            max_len = 2
            if len(bounds_to_plot) == max_len:
                row = 2 if bound == "lower" else 3
                row_mtr0 = row
                row_mtr1 = row
                col_mtr0 = 1
                col_mtr1 = 2
            else:
                col_mtr0 = 1
                col_mtr1 = 1
                row_mtr0 = 2
                row_mtr1 = 3

            for j in range(k_bernstein + 1):
                fig.add_trace(
                    go.Scatter(
                        x=data["beta_late"],
                        y=data[f"x_{j}_{bound}"],
                        mode="lines",
                        name=f"{j}",
                        legendgroup="MTR0",
                        legendgrouptitle={"text": "Solution: Basis Coefficients"},
                        line={"color": color_by_x_number[j]},
                    ),
                    row=row_mtr0,
                    col=col_mtr0,
                )

                fig.add_trace(
                    go.Scatter(
                        x=data["beta_late"],
                        y=data[f"x_{j + k_bernstein + 1}_{bound}"],
                        mode="lines",
                        name=f"x_{j + k_bernstein + 1}",
                        legendgroup="MTR1",
                        legendgrouptitle={"text": "Coefs on MTR d = 1"},
                        line={"color": color_by_x_number[j]},
                        showlegend=False,
                    ),
                    row=row_mtr1,
                    col=col_mtr1,
                )

        fig.update_xaxes(matches="x")

        # Make less wide
        fig.update_layout(width=800, height=800)

        # Add vertical lines at the kinks to all subplots
        for beta in kinks_beta:
            fig.add_shape(
                {
                    "type": "line",
                    "x0": beta,
                    "y0": 0,
                    "x1": beta,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",
                    "line": {"color": "rgba(0, 0, 0, 0.1)", "width": 1},
                },
            )

        subtitle = (
            f"<br><sup>Bernstein Polynomial of Degree {k_bernstein},"
            f"{num_grid_points} Grid Points</sup>"
        )

        fig.update_layout(
            title_text=(
                "Identification Bounds and Basis Function Coefficients" + subtitle
            ),
        )

        # Put legend below plot
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        # Adjust x-axis range for 2nd and 3rd row (MTR coefs are bounded between 0, 1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_yaxes(range=[0, 1], row=3, col=1)

        # Turn off legend
        if not show_legend:
            fig.update_layout(showlegend=False)

        fig.write_image(path_to_plot)
