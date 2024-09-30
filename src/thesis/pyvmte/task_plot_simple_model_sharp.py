"""Plot the simple model for a specification resulting in sharp bounds."""

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product, task

from thesis.config import BLD


# --------------------------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------------------------
class _Arguments(NamedTuple):
    param_to_vary: str
    path_to_plot: Annotated[Path, Product]
    shape_constraint: tuple[str, str] = ("decreasing", "decreasing")
    path_to_data: Path = BLD / "data" / "solutions_simple_model_sharp.pkl"


ID_TO_KWARGS = {
    "y1_c": _Arguments(
        param_to_vary="y1_c",
        path_to_plot=BLD / "figures" / "simple_model_sharp_by_y1_ct.png",
    ),
    "y0_c": _Arguments(
        param_to_vary="y0_c",
        path_to_plot=BLD / "figures" / "simple_model_sharp_by_y0_ct.png",
    ),
}

for kwargs in ID_TO_KWARGS.values():

    @task(kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_simple_model_solution_sharp(
        param_to_vary: str,
        path_to_plot: Annotated[Path, Product],
        shape_constraint: tuple[str, str],
        path_to_data: Path,
    ) -> None:
        """Plot solution to the simple model over a grid of parameter values."""
        # Solve the program for a range of y1_c values from 0 to 1

        results = pd.read_pickle(path_to_data)  # noqa: S301

        results = results[results["shape_constraint"] == shape_constraint]

        y1_at = results["y1_at"][0]
        y0_at = results["y0_at"][0]
        y1_nt = results["y1_nt"][0]
        y0_nt = results["y0_nt"][0]

        # Get value of y1_c and y0_c closest to 0.5
        if param_to_vary == "y1_c":
            _idx_closest = results["y0_c"].sub(0.5).abs().idxmin()
            _closest = results["y0_c"][_idx_closest]
            results = results[results["y0_c"] == _closest]

        elif param_to_vary == "y0_c":
            _idx_closest = results["y1_c"].sub(0.5).abs().idxmin()
            _closest = results["y1_c"][_idx_closest]
            results = results[results["y1_c"] == _closest]

        # ------------------------------------------------------------------------------
        # Plot the results
        # ------------------------------------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results[param_to_vary],
                y=results["upper_bound"],
                mode="lines",
                name="Upper bound",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=results[param_to_vary],
                y=results["lower_bound"],
                mode="lines",
                name="Lower bound",
            ),
        )

        # Add nan as red crosses
        _idx_nan = results["upper_bound"].isna()
        fig.add_trace(
            go.Scatter(
                x=results[param_to_vary][_idx_nan],
                y=results[param_to_vary][_idx_nan] * 0,
                mode="markers",
                marker={"color": "red", "symbol": "x", "opacity": 0.1},
                name="Empty identified set",
            ),
        )

        fig.update_layout(
            title=(
                f"Bounds on the LATE for different values of {param_to_vary}"
                f"<br><sup>Other at 0.5, y1 at = {y1_at}, "
                f"y0 at = {y0_at}, y1 nt = {y1_nt}, y0 nt = {y0_nt}<br><sup>"
            ),
            xaxis_title=f"{param_to_vary}",
            yaxis_title="Bounds on Target",
        )

        fig.write_image(path_to_plot)


# --------------------------------------------------------------------------------------
# Plot for changing both parameters simultaneously
# --------------------------------------------------------------------------------------


def task_plot_simple_model_solution_sharp_both(
    path_to_plot_upper: Annotated[Path, Product] = BLD
    / "figures"
    / "simple_model_sharp_by_y0_y1_ct_upper.png",
    path_to_plot_lower: Annotated[Path, Product] = BLD
    / "figures"
    / "simple_model_sharp_by_y0_y1_ct_lower.png",
    path_to_data: Path = BLD / "data" / "solutions_simple_model_sharp.pkl",
) -> None:
    """Plot solution to the simple model over a meshgrid of parameter values."""
    results = pd.read_pickle(path_to_data)  # noqa: S301

    y1_at = results["y1_at"][0]
    y0_at = results["y0_at"][0]
    y1_nt = results["y1_nt"][0]
    y0_nt = results["y0_nt"][0]

    # ----------------------------------------------------------------------------------
    # Plot the results
    # ----------------------------------------------------------------------------------

    for bound in ["upper_bound", "lower_bound"]:
        fig = go.Figure()

        _idx_nan = results[bound].isna()

        # Make scatter with coloring by res_upper but only for non-nan values
        fig.add_trace(
            go.Scatter(
                x=results["y1_c"][~_idx_nan],
                y=results["y0_c"][~_idx_nan],
                mode="markers",
                marker={
                    "color": results[bound][~_idx_nan],
                    "colorscale": "Viridis",
                    "colorbar": {
                        "title": f"ID set: {bound.replace('_', ' ').capitalize()}",
                    },
                },
                name="Solution region",
            ),
        )

        # Plot the nan values
        fig.add_trace(
            go.Scatter(
                x=results["y1_c"][_idx_nan],
                y=results["y0_c"][_idx_nan],
                mode="markers",
                marker={"color": "red", "symbol": "x", "opacity": 0.1},
                name="Empty identified set",
            ),
        )

        fig.update_layout(
            legend={
                "x": 0.8,  # Position the legend at the right side
                "y": -0.2,  # Center the legend vertically
                "traceorder": "normal",
            },
        )

        # Title the axes
        fig.update_layout(
            title=(
                f"Bounds on the LATE for different values of complier outcomes"
                f"<br><sup>Decreasing MTR functions. y1 at = {y1_at}, "
                f"y0 at = {y0_at}, y1 nt = {y1_nt}, y0 nt = {y0_nt}<br><sup>"
            ),
            xaxis_title="$E[Y(1)|complier]$",
            yaxis_title="$E[Y(0)|complier]$",
        )

        if bound == "upper_bound":
            fig.write_image(path_to_plot_upper)
        else:
            fig.write_image(path_to_plot_lower)
