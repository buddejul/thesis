"""Run simulations for Imbens and Manski (2004) ECMA."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.imbens_manski._task_sim_imbens_manski import (
    ID_TO_KWARGS,
    N_BOOT,
    N_SIMS,
    _Arguments,
)

PATHS_PLOTS = {
    "analytical": BLD / "imbens_manski" / "imbens_manski_analytical.png",
    "bootstrap": BLD / "imbens_manski" / "imbens_manski_bootstrap.png",
}


def task_plot_sim_results(
    id_to_kwargs: dict[str, _Arguments] = ID_TO_KWARGS,
    paths_plots: Annotated[dict[str, Path], Product] = PATHS_PLOTS,
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

    alpha = data["alpha"].unique()

    # Compute mean coverage probability
    cols_groupby = ["p", "n_obs", "ci_type", "alpha"]

    mean_coverage_probability = data.groupby(cols_groupby).mean().reset_index()

    n_obs_to_line_type = {100: "solid", 250: "dash", 1000: "dot"}

    for ci_type in ["analytical", "bootstrap"]:
        fig = go.Figure()

        for n_obs in data["n_obs"].unique():
            query = f"n_obs == {n_obs} and ci_type == '{ci_type}'"
            sub_df = mean_coverage_probability.query(query)

            fig.add_trace(
                go.Scatter(
                    x=sub_df["p"],
                    y=sub_df["ci_idset_covers"],
                    mode="lines+markers",
                    name=f"n_obs = {n_obs}",
                    legendgroup="idset_ci",
                    legendgrouptitle={"text": "CI for Identified Set"},
                    line_dash=n_obs_to_line_type.get(n_obs, "solid"),
                    line={"color": "blue"},
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
                    line_dash=n_obs_to_line_type.get(n_obs, "solid"),
                    line={"color": "red"},
                ),
            )

        fig.update_layout(
            title=(
                "CI for Parameter is not Uniformly Valid over p"
                f"<br><sup> CI Type: {ci_type}, alpha: {alpha}, "
                f"Number of simulations: {N_SIMS} </sup>"
            ),
            xaxis_title="p",
            yaxis_title="Coverage probability",
        )

        if ci_type == "bootstrap":
            fig.add_annotation(
                x=1,
                y=-0.1,
                xref="paper",
                yref="paper",
                text=f"Number of bootstrap samples: {N_BOOT}",
                showarrow=False,
            )

        fig.write_image(paths_plots[ci_type])
