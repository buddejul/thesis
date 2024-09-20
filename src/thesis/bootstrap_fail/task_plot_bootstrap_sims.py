"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.bootstrap_fail.task_bootstrap_sims import ID_TO_KWARGS, _Arguments
from thesis.config import BLD


def task_plot_boostrap_sims(
    id_to_kwargs: dict[str, _Arguments] = ID_TO_KWARGS,
    path_to_plot_coverage: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "coverage.png",
    ),
    path_to_plot_length: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "length.png",
    ),
) -> None:
    """Plot the coverage probability of the confidence interval."""
    # Read all the bootstrap result and for each calculate the coverage probability
    # of the confidence interval. Then plot against u_hi.
    coverage = []

    for kwargs in id_to_kwargs.values():
        res = pd.read_pickle(kwargs.path_to_data)  # noqa: S301
        res["ci_covers"] = (res["lo"] < res["true"]) & (res["hi"] > res["true"])
        res["ci_length"] = res["hi"] - res["lo"]
        coverage.append(
            (
                kwargs.u_hi,
                kwargs.n_obs,
                res["ci_covers"].mean(),
                res["ci_length"].mean(),
            ),
        )

    data = pd.DataFrame(coverage, columns=["u_hi", "n_obs", "coverage", "length"])

    fig = go.Figure()
    for n_obs in data.n_obs.unique():
        data_sub = data[data.n_obs == n_obs]
        fig.add_trace(
            go.Scatter(x=data_sub.u_hi, y=data_sub.coverage, name=f"n_obs={n_obs}"),
        )

    fig.update_layout(
        title="Coverage probability of confidence interval",
        xaxis_title="u_hi",
        yaxis_title="Coverage probability",
    )

    fig.write_image(path_to_plot_coverage)

    fig = go.Figure()
    for n_obs in data.n_obs.unique():
        data_sub = data[data.n_obs == n_obs]
        fig.add_trace(
            go.Scatter(x=data_sub.u_hi, y=data_sub.length, name=f"n_obs={n_obs}"),
        )

    fig.update_layout(
        title="Average length of confidence interval",
        xaxis_title="u_hi",
        yaxis_title="Average length",
    )

    fig.write_image(path_to_plot_length)
