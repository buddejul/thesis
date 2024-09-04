"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.simple_model.task_simple_model_sims import ID_TO_KWARGS, _Arguments


# TODO(@buddejul): Put graphs below in a loop, currently there is a lot of copy/paste.
def task_plot_boostrap_sims(
    id_to_kwargs: dict[str, _Arguments] = ID_TO_KWARGS,
    path_to_plot_coverage: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "coverage.png",
    ),
    path_to_plot_length: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "length.png",
    ),
    path_to_plot_means: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "means.png",
    ),
) -> None:
    """Plot the coverage probability of the confidence interval."""
    # Read all the bootstrap result and for each calculate the coverage probability
    # of the confidence interval. Then plot against bootstrap_method and lates complier.
    coverage = []

    for kwargs in id_to_kwargs.values():
        res = pd.read_pickle(kwargs.path_to_data)  # noqa: S301
        res["ci_covers"] = (res["lo"] < res["true"]) & (res["hi"] > res["true"])
        res["ci_length"] = res["hi"] - res["lo"]
        coverage.append(
            (
                kwargs.u_hi,
                kwargs.n_obs,
                kwargs.local_ates.complier,
                kwargs.bootstrap_method,
                kwargs.constraint_mtr,
                res["true"].mean(),
                res["ci_covers"].mean(),
                res["ci_length"].mean(),
                res["lo"].mean(),
                res["hi"].mean(),
            ),
        )

    cols = [
        "u_hi",
        "n_obs",
        "late_complier",
        "bootstrap_method",
        "constraint_mtr",
        "true",
        "coverage",
        "length",
        "ci_lo",
        "ci_hi",
    ]

    data = pd.DataFrame(coverage, columns=cols)

    # TODO(@buddejul): Do this properly and loop over mtr constraints.
    data = data[data.constraint_mtr == "increasing"]

    fig = go.Figure()
    for n_obs in data.n_obs.unique():
        for bootstrap_method in ["standard", "numerical_delta"]:
            data_sub = data[
                (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
            ]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.coverage,
                    name=f"n_obs={n_obs}, <br> {bootstrap_method}",
                ),
            )

    fig.update_layout(
        title="Coverage probability of CI for true parameter",
        xaxis_title="late_complier",
        yaxis_title="Coverage probability",
    )

    fig.write_image(path_to_plot_coverage)

    fig = go.Figure()
    for n_obs in data.n_obs.unique():
        for bootstrap_method in ["standard", "numerical_delta"]:
            data_sub = data[
                (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
            ]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.length,
                    name=f"n_obs={n_obs}, <br> {bootstrap_method}",
                ),
            )

    fig.update_layout(
        title="Average length of confidence interval",
        xaxis_title="late_complier",
        yaxis_title="Average length",
    )

    fig.write_image(path_to_plot_length)

    # ==================================================================================
    # Plot Means
    # ==================================================================================
    fig = go.Figure()
    for n_obs in data.n_obs.unique():
        for bootstrap_method in ["standard", "numerical_delta"]:
            data_sub = data[
                (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
            ]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.ci_lo,
                    name=f"Upper, n_obs={n_obs}, {bootstrap_method}",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.ci_hi,
                    name=f"Lower, n_obs={n_obs}, {bootstrap_method}",
                ),
            )

    fig.update_layout(
        title="Means of CI Bounds",
        xaxis_title="late_complier",
        yaxis_title="Mean CI Bounds",
    )

    fig.write_image(path_to_plot_means)
