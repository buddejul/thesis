"""Tasks to plot results from relaxation bootstrap."""
from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD
from thesis.misc.task_relaxation_bootstrap import ID_TO_KWARGS

RES_FILES = [val.path_to_results for key, val in ID_TO_KWARGS.items()]


@pytask.mark.relax_boot
def task_combine_relaxation_bootstrap_results(
    res_files: list[Path] = RES_FILES,
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "relaxation_bootstrap" / "relaxation_bootstrap_combined.pkl"
    ),
) -> None:
    """Combine results from relaxation bootstrap tasks."""
    dfs = [pd.read_pickle(file) for file in res_files]

    out = pd.concat(dfs, ignore_index=True)

    out.to_pickle(path_to_combined)


@pytask.mark.relax_boot
def task_plot_coverage_by_method(
    path_to_combined: Path = BLD
    / "data"
    / "relaxation_bootstrap"
    / "relaxation_bootstrap_combined.pkl",
    path_to_plot: Annotated[Path, Product] = BLD
    / "figures"
    / "relaxation_bootstrap"
    / "coverage_by_method.html",
) -> None:
    """Plot coverage by method and slope parameter."""
    combined = pd.read_pickle(path_to_combined)

    data = combined.groupby(["method", "slope"]).mean().reset_index()

    fig = go.Figure()

    for method in data.method.unique():
        sub_data = data[data.method == method]

        fig.add_trace(
            go.Scatter(
                y=sub_data["covers_lower_one_sided"],
                x=sub_data["slope"],
                mode="lines+markers",
                name=f"{method.replace('_', ' ').capitalize()}",
            ),
        )

    fig.update_layout(
        title="Coverage by Method and Parameter",
        xaxis_title="Slope",
        yaxis_title="Coverage",
    )

    # Add note: Data is Normal with sigma = 1
    fig.add_annotation(
        text="Data: Normal(, 1)",
        xref="paper",
        yref="paper",
        x=1,
        y=-0.1,
        showarrow=False,
    )

    fig.write_html(path_to_plot)
