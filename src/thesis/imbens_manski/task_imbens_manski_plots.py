"""Run simulations for Imbens and Manski (2004) ECMA."""

from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.imbens_manski.task_imbens_manski_sims import (
    ID_TO_KWARGS,
    N_OBS_VALUES,
)


class _Arguments(NamedTuple):
    p: float
    n_obs: int
    n_sims: int
    path_to_results: Path
    target: str = "idset"


PATHS_TO_RESULTS = [arg.path_to_results for arg in ID_TO_KWARGS.values()]


def task_plot_sim_results(
    path_to_sim_results: list[Path] = PATHS_TO_RESULTS,
    path_to_plot: Annotated[Path, Product] = BLD / "imbens_manski_sims.png",
) -> None:
    """Plot simulation results."""
    # Combine all results
    datasets = []
    for path in path_to_sim_results:
        _df = pd.read_pickle(path)  # noqa: S301
        p = float(path.stem.split("_")[1])
        n_obs = int(path.stem.split("_")[4])
        _df["p"] = p
        _df["n_obs"] = n_obs
        datasets.append(_df)
    data = pd.concat(datasets)

    # Compute mean coverage probability
    mean_coverage_probability = data.groupby(["p", "n_obs"]).mean().reset_index()

    # Plot
    fig = go.Figure()
    for n_obs in N_OBS_VALUES:
        sub_df = mean_coverage_probability[mean_coverage_probability.n_obs == n_obs]
        fig.add_trace(
            go.Scatter(
                x=sub_df["p"],
                y=sub_df["ci_idset_covers"],
                mode="lines+markers",
                name=f"n_obs = {n_obs}",
                legendgroup="idset_ci",
                legendgrouptitle={"text": "CI for Identified Set"},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=sub_df["p"],
                y=sub_df["ci_param_covers"],
                mode="lines+markers",
                name=f"n_obs = {n_obs}",
                legendgroup="param_ci",
                legendgrouptitle={"text": "CI for Parameter"},
            ),
        )

    fig.update_layout(
        title="CI for Parameter is not Uniformly Valid over p",
        xaxis_title="p",
        yaxis_title="Coverage probability",
    )
    fig.write_image(path_to_plot)
