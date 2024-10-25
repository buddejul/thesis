"""Plot relaxation of MTE problem to convex problem with larger parameter space."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from thesis.config import BLD

# Build matrices from scratch, then optimize.
# For the non-linear one, add the unit-ball constraint
from thesis.misc.task_compute_relaxation_mte import ID_TO_KWARGS

path_to_results = [val.path_to_data for val in ID_TO_KWARGS.values()]


class _Arguments(NamedTuple):
    path_to_results: list[Path]
    path_to_plot: Annotated[Path, Product]
    path_to_plot_html: Annotated[Path, Product]
    k_bernstein: int


KWARGS = {
    "k_bernstein_11": _Arguments(
        path_to_results=path_to_results,
        path_to_plot=BLD
        / "figures"
        / "relaxation"
        / "relaxation_mte_k_bernstein_11.png",
        path_to_plot_html=BLD
        / "figures"
        / "relaxation"
        / "relaxation_mte_k_bernstein_11.html",
        k_bernstein=11,
    ),
}

for id_, kwargs in KWARGS.items():

    @pytask.mark.relax_wip()
    @task(name=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_relaxation_mte(
        path_to_results: list[Path],
        path_to_plot: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        k_bernstein: int,
    ) -> None:
        """Task for solving original and relaxed convex problem."""
        del k_bernstein  # For linting.

        # Load results from identification task
        results = []
        for path in path_to_results:
            with Path.open(path, "rb") as file:
                results.append(pickle.load(file))

        # Merge the dataframes
        data = pd.concat(results)

        # Plot the results by k_approximation

        fig = go.Figure()

        for k_approximation in data["k_approximation"].unique():
            data_k = data[data["k_approximation"] == k_approximation].reset_index(
                names="beta",
            )

            if k_approximation == data["k_approximation"].unique()[0]:
                fig.add_trace(
                    go.Scatter(
                        x=data_k["beta"],
                        y=data_k["lp"],
                        mode="lines",
                        name="LP Solution",
                        legendgroup="lp",
                    ),
                )

            fig.add_trace(
                go.Scatter(
                    x=data_k["beta"],
                    y=data_k["convex"],
                    mode="lines",
                    name=f"Convex, k = {k_approximation}",
                    legendgroup=f"k = {k_approximation}",
                ),
            )

        fig.update_layout(
            title="Relaxation of MTE Problem to Convex Problem",
            xaxis_title="Beta",
            yaxis_title="Value",
            legend_title="",
        )

        fig.write_image(path_to_plot)
        fig.write_html(path_to_plot_html)


# def task_plot_relaxation_mte() -> None:
#     """Task for plotting the relaxation of the MTE problem."""
#         result_polynomial_degree4,


#     fig.add_trace(
#         go.Scatter(
#         ),

#     for constraint in ["unit_ball", "polynomial_degree4"]:
#         fig.add_trace(
#             go.Scatter(
#             ),
