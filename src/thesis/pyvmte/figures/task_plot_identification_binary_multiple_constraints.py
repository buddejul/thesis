"""Plot solutions to binary IV model (multiple constraints simultaneously)."""

# TODO(@buddejul): Should make sure, that using constant spline MTRs as the DGP does not
# introduce the kinks we observe for the Bernstein solution.
# Alternative approach: Could we get the lower/upper bound Bernstein MTR from the
# identification function and then take some kind of average?

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from thesis.config import BLD

# --------------------------------------------------------------------------------------
# Plot with Multiple shape restrictions, without solutions
# --------------------------------------------------------------------------------------
# What we want
# idestimands: late, sharp
# Then in a single plot:
# - Constant splines: No constraint, MTE decreasing, Positive treatment response
# - Bernstein, k = 11: No constraint, MTE decreasing, Positive treatment response
# - Bernstein, k = 3: No constraint, MTE decreasing, Positive treatment response
# Legend groups are: Constant splines, Bernstein k = 11, Bernstein k = 3


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    path_to_plot_html: Annotated[Path, Product]
    path_to_solutions: Path
    idestimands: str


ID_TO_KWARGS = {
    f"multiple_constraints_{idestimands}": _Arguments(
        path_to_plot=BLD
        / "figures"
        / "binary_iv"
        / f"multiple_constraints_{idestimands}.png",
        path_to_plot_html=BLD
        / "figures"
        / "binary_iv"
        / "html"
        / f"multiple_constraints_{idestimands}.html",
        path_to_solutions=BLD
        / "data"
        / "solutions"
        / "full_solutions_simple_model_combined.pkl",
        idestimands=idestimands,
    )
    for idestimands in ["late", "sharp"]
}

for id_, kwargs in ID_TO_KWARGS.items():  # type: ignore[assignment]

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.plot_for_paper_wip
    def task_plot_identification_binary_iv_multiple_constraints(
        path_to_plot: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        path_to_solutions: Path,
        idestimands: str,
    ) -> None:
        """Plot solutions to binary IV model (multiple constraints simultaneously)."""
        # Load solutions
        data = pd.read_pickle(path_to_solutions)

        # Filter for the correct idestimands
        data = data[data["idestimands"] == idestimands]

        # Sort by complier_late
        data = data.sort_values("complier_late")

        # Drop all columns "x_"
        data = data.drop(columns=[col for col in data.columns if col.startswith("x_")])

        fig = go.Figure()

        color_by_bfunc_type = {
            "constant": "blue",
            "bernstein_k_2": "red",
            "bernstein_k_3": "purple",
            "bernstein_k_11": "orange",
        }

        dash_by_constraint_type = {
            "none": "solid",
            "mte_monotone": "dash",
            "monotone_response": "dot",
        }

        constraint_type_to_legend_name = {
            "none": "None",
            "mte_monotone": "MTE Decreasing",
            "monotone_response": "Positive Treatment Response",
        }

        # Add constant splines
        for constraint_type in ["none", "mte_monotone", "monotone_response"]:
            data_plot = data[data["constraint_type"] == constraint_type]

            data_plot = data_plot[data_plot["bfunc_type"] == "constant"]

            assert data_plot.index.is_unique

            data_plot = data_plot.reset_index()

            for bound in ["lower", "upper"]:
                show_legend = bound == "lower"

                fig.add_trace(
                    go.Scatter(
                        x=data_plot["complier_late"],
                        y=data_plot[f"{bound}_bound"],
                        mode="lines",
                        legendgroup=f"constant_{constraint_type}",
                        # },
                        name=(
                            f"Constant: "
                            f"{constraint_type_to_legend_name[constraint_type]}"
                        ),
                        showlegend=show_legend,
                        line={
                            "color": color_by_bfunc_type["constant"],
                            "dash": dash_by_constraint_type[constraint_type],
                        },
                    ),
                )

        # Add Bernstein
        for k_bernstein in [11, 3, 2]:
            for constraint_type in ["none", "mte_monotone", "monotone_response"]:
                data_plot = data[data["constraint_type"] == constraint_type]
                data_plot = data_plot[data_plot["bfunc_type"] == "bernstein"]
                data_plot = data_plot[data_plot["k_bernstein"] == k_bernstein]

                assert data_plot.index.is_unique

                data_plot = data_plot.reset_index()

                for bound in ["lower", "upper"]:
                    show_legend = bound == "lower"

                    fig.add_trace(
                        go.Scatter(
                            x=data_plot["complier_late"],
                            y=data_plot[f"{bound}_bound"],
                            mode="lines",
                            name=f"Bernstein k = {k_bernstein}: {constraint_type}",
                            legendgroup=(
                                f"Bernstein k = {k_bernstein},"
                                f" {constraint_type_to_legend_name[constraint_type]}"
                            ),
                            showlegend=show_legend,
                            line={
                                "color": color_by_bfunc_type[
                                    f"bernstein_k_{k_bernstein}"
                                ],
                                "dash": dash_by_constraint_type[constraint_type],
                            },
                        ),
                    )

        fig.update_layout(
            title_text=(
                f"Identification Bounds for Binary IV Model: {idestimands.capitalize()}"
            ),
        )

        # x-title
        fig.update_xaxes(title_text="Complier Late")

        fig.write_image(path_to_plot)
        fig.write_html(path_to_plot_html)
