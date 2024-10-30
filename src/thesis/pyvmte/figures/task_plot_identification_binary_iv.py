"""Plot solutions to binary IV model showing the solution along the bounds."""

# TODO(@buddejul): Should make sure, that using constant spline MTRs as the DGP does not
# introduce the kinks we observe for the Bernstein solution.
# Alternative approach: Could we get the lower/upper bound Bernstein MTR from the
# identification function and then take some kind of average?

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from pytask import Product, task

from thesis.config import BLD
from thesis.pyvmte.solutions.task_solve_simple_model import num_gridpoints


def _generate_color_by_x_number(num_bfuncs: int) -> dict[int, str]:
    # Cycle through the palette to create the color dictionary
    # Cycle through red numbers using rgb
    return {i: f"rgb({255 - i * 255 // num_bfuncs}, 0, 0)" for i in range(num_bfuncs)}


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    k_bernstein: int
    bfunc_type: str
    constraint_type: str
    idestimands: str
    bounds_to_plot: tuple[str] = ("lower",)
    path_to_solutions: Path = (
        BLD / "data" / "solutions" / "full_solutions_simple_model_combined.pkl"
    )
    path_to_plot_html: Annotated[Path, Product] | None = None


ID_TO_KWARGS = {
    f"degree_{bfunc_type}_{k}_{constraint_type}_{idestimands}": _Arguments(
        path_to_plot=BLD
        / "figures"
        / "binary_iv"
        / f"id_{bfunc_type}_{k}_{constraint_type}_{idestimands}.png",
        k_bernstein=k,
        bfunc_type=bfunc_type,
        constraint_type=constraint_type,
        idestimands=idestimands,
    )
    for bfunc_type in ["bernstein", "constant"]
    for k in ([2, 3, 11] if bfunc_type == "bernstein" else [np.nan])  # type: ignore[list-item]
    for constraint_type in [
        "none",
        "shape_constraints",
        "mte_monotone",
        "monotone_response",
    ]
    for idestimands in ["late", "sharp"]
}

# Add path_to_html for all tasks
for id_, kwargs in ID_TO_KWARGS.items():
    ID_TO_KWARGS[id_] = kwargs._replace(
        path_to_plot_html=BLD
        / "figures"
        / "binary_iv"
        / "html"
        / (
            f"id_{kwargs.bfunc_type}_{kwargs.k_bernstein}_"
            f"{kwargs.constraint_type}_{kwargs.idestimands}.html"
        ),
    )

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.plot_for_paper
    def task_plot_identification_binary_iv(  # noqa: PLR0915, C901, PLR0912
        path_to_plot: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        bounds_to_plot: tuple[str],
        k_bernstein: int,
        path_to_solutions: Path,
        bfunc_type: str,
        constraint_type: str,
        idestimands: str,
        show_legend: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Plot bounds and bfunc coefs for binary IV model."""
        num_bfuncs_const = 2 * 4
        num_bfuncs = (
            2 * (k_bernstein + 1) if bfunc_type == "bernstein" else num_bfuncs_const
        )

        # ------------------------------------------------------------------------------
        # Load solutions
        # ------------------------------------------------------------------------------
        data = pd.read_pickle(path_to_solutions)

        data = data[data["bfunc_type"] == bfunc_type]
        data = data[data["constraint_type"] == constraint_type]
        data = data[data["idestimands"] == idestimands]
        if bfunc_type == "bernstein":
            data = data[data["k_bernstein"] == k_bernstein]

        assert data.index.is_unique
        data = data.reset_index()

        data = data.sort_values("complier_late")

        # ------------------------------------------------------------------------------
        # Identify kinks
        # ------------------------------------------------------------------------------

        # Identify kinks in "lower_bound" by looking at changes in the first
        # derivative. First, approximate the derivative by taking differences
        b_late_grid = np.linspace(-1, 1, num_gridpoints)

        data["d_lower_bound"] = np.gradient(
            data["lower_bound"].to_numpy(),
            b_late_grid,
        )

        data["d_lower_bound"]

        # Find the kinks by looking at changes in the first derivative larger than
        # some epsilon Compute differences in the first derivative
        data["dd_lower_bound"] = np.gradient(data["d_lower_bound"], b_late_grid)

        # Find the kinks
        epsilon = 0.1
        kinks = data.loc[
            (data["dd_lower_bound"].abs() > epsilon),
            ["complier_late", "lower_bound", "d_lower_bound", "dd_lower_bound"],
        ]

        kinks_beta = kinks["complier_late"].to_numpy()

        # --------------------------------------------------------------------------
        # Figure
        # --------------------------------------------------------------------------

        # Example usage
        color_by_x_number = _generate_color_by_x_number(num_bfuncs)

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
                    x=data["complier_late"],
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

            for j in range(num_bfuncs // 2):
                fig.add_trace(
                    go.Scatter(
                        x=data["complier_late"],
                        y=data[f"x_{bound}_{j}"],
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
                        x=data["complier_late"],
                        y=data[f"x_{bound}_{j + num_bfuncs // 2}"],
                        mode="lines",
                        name=f"x_{j + num_bfuncs // 2}",
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

        types_to_subtitles = {
            "none": "None",
            "shape_constraints": "Decreasing MTR Functions",
            "mte_monotone": "MTE Decreasing",
            "monotone_response": "Positive Treatment Response",
        }

        subtitle = (
            f"<br><sup>Bernstein Polynomial of Degree {k_bernstein}, "
            f"{num_gridpoints} Grid Points"
            f"<br>Shape restriction: {types_to_subtitles[constraint_type]}</sup>"
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
        fig.update_yaxes(range=[-0.1, 1.1], row=2, col=1)
        fig.update_yaxes(range=[-0.1, 1.1], row=3, col=1)

        # Turn off legend
        if not show_legend:
            fig.update_layout(showlegend=False)

        fig.write_image(path_to_plot)

        if path_to_plot_html is not None:
            fig.write_html(path_to_plot_html)
