"""Plot the true solution for the simple model setting."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product, task

from thesis.config import BLD
from thesis.simple_model.funcs import _idset


class _Arguments(NamedTuple):
    constraint_mtr: str
    path_to_plot: Annotated[Path, Product]
    u_hi: float = 0.2
    grid_points: int = 250
    endpoints: tuple[float, float] = (-0.1, 0.1)
    pscores: tuple[float, float] = (0.4, 0.6)


constr_mtr_to_plot = ["none", "increasing"]

ID_TO_KWARGS = {
    f"plot_constraint_mtr_{constraint_mtr}": _Arguments(
        constraint_mtr=constraint_mtr,
        path_to_plot=Path(
            BLD / "boot" / "figures" / f"solution_constraint_mtr_{constraint_mtr}.png",
        ),
    )
    for constraint_mtr in constr_mtr_to_plot
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_solution(
        u_hi: float,
        grid_points: int,
        endpoints: tuple[float, float],
        pscores: tuple[float, float],
        constraint_mtr: str,
        path_to_plot: Annotated[Path, Product],
    ) -> None:
        """Plot the solution for the simple model."""
        beta_grid = np.linspace(endpoints[0], endpoints[1], grid_points)

        solutions = np.zeros((grid_points, 2))

        for i, beta in enumerate(beta_grid):
            solutions[i, :] = _idset(
                b_late=beta,
                u_hi=u_hi,
                pscores_hat=pscores,
                constraint_mtr=constraint_mtr,
            )

        data = pd.DataFrame(solutions, columns=["lower_bound", "upper_bound"])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=beta_grid,
                y=data["lower_bound"],
                mode="lines",
                name="Lower Bound",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=beta_grid,
                y=data["upper_bound"],
                mode="lines",
                name="Upper Bound",
            ),
        )

        fig.update_layout(
            title=(
                "Solution for the Simple Model: "
                f"MTR Constraint = {constraint_mtr.capitalize()}"
            ),
            xaxis_title="Beta",
            yaxis_title="Value",
        )

        # Add a note with all parameters
        fig.add_annotation(
            text=f"u_hi = {u_hi}, pscores = {pscores}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
        )

        fig.write_image(path_to_plot)
