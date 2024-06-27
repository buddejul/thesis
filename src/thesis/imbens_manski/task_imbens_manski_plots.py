"""Run simulations for Imbens and Manski (2004) ECMA."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD
from thesis.imbens_manski.task_imbens_manski_sims import (
    ID_TO_KWARGS,
    _Arguments,
)


@pytask.mark.skip()
def task_plot_sim_results(
    id_to_kwargs: dict[str, _Arguments] = ID_TO_KWARGS,
    path_to_plot: Annotated[Path, Product] = BLD / "imbens_manski_sims.png",
) -> None:
    """Plot simulation results."""
    # Combine all results
    datasets = []
    for res in id_to_kwargs.values():
        _df = pd.read_pickle(res.path_to_results)  # noqa: S301
        _df["p"] = res.p
        _df["n_obs"] = res.n_obs
        _df["ci_type"] = res.ci_type
        _df["alpha"] = res.alpha
        datasets.append(_df)

    data = pd.concat(datasets)

    # Compute mean coverage probability
    # TODO: This is dangerous since we have to manually specify the columns.

    # Get all columns except "ci_idset_covers" and "ci_param_covers"
    cols_groupby = data.columns[
        ~data.columns.isin(["ci_idset_covers", "ci_param_covers"])
    ]

    mean_coverage_probability = data.groupby(cols_groupby).mean().reset_index()

    # Plot
    fig = go.Figure()
    for n_obs in data["n_obs"].unique():
        for ci_type in ["analytical", "bootstrap"]:
            query = f"n_obs == {n_obs} and ci_type == '{ci_type}'"
            sub_df = mean_coverage_probability.query(query)

            fig.add_trace(
                go.Scatter(
                    x=sub_df["p"],
                    y=sub_df["ci_idset_covers"],
                    mode="lines+markers",
                    name=f"n_obs = {n_obs}, {ci_type}",
                    legendgroup="idset_ci",
                    legendgrouptitle={"text": "CI for Identified Set"},
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_df["p"],
                    y=sub_df["ci_param_covers"],
                    mode="lines+markers",
                    name=f"n_obs = {n_obs}, {ci_type}",
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
