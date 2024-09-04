"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.simple_model.task_simple_model_sims import ID_TO_KWARGS, _Arguments


# TODO(@buddejul): Put graphs below in a loop, currently there is a lot of copy/paste.
# TODO(@buddejul): Include true parameters in plots.
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
        # Note we compute coverage for the true parameter. Not coverage for the ID set.
        res["ci_covers_true_param"] = (res["lo"] <= res["true"]) & (
            res["hi"] >= res["true"]
        )
        res["ci_length"] = res["hi"] - res["lo"]
        coverage.append(
            (
                kwargs.u_hi,
                kwargs.n_obs,
                kwargs.local_ates.complier,
                kwargs.bootstrap_method,
                kwargs.constraint_mtr,
                res["true"].mean(),
                res["ci_covers_true_param"].mean(),
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

    data = data.sort_values("late_complier")

    color_by_bootstrap_method = {
        "standard": "blue",
        "numerical_delta": "red",
    }

    line_type_by_n_obs = {
        250: "solid",
        1_000: "dash",
    }

    for col_to_plot in ["coverage", "length"]:
        fig = go.Figure()
        for n_obs in data.n_obs.unique():
            for bootstrap_method in ["standard", "numerical_delta"]:
                data_sub = data[
                    (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=data_sub.late_complier,
                        y=data_sub[col_to_plot],
                        name=f"n_obs={n_obs}",
                        legendgroup=f"{bootstrap_method}",
                        legendgrouptitle_text=(
                            f"{bootstrap_method.replace('_', ' ').capitalize()}"
                            "Bootstrap"
                        ),
                        line={
                            "color": color_by_bootstrap_method[bootstrap_method],
                            "dash": line_type_by_n_obs[n_obs],
                            "width": 2,
                        },
                    ),
                )

        fig.update_layout(
            title=f"{col_to_plot.capitalize()} of CI for true parameter",
            xaxis_title="LATE Complier",
            yaxis_title=f"{col_to_plot.capitalize()}",
        )

        if col_to_plot == "coverage":
            path_to_plot = path_to_plot_coverage
        elif col_to_plot == "length":
            path_to_plot = path_to_plot_length

        fig.write_image(path_to_plot)

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
                    legendgroup=f"{bootstrap_method}",
                    legendgrouptitle_text=(
                        f"{bootstrap_method.replace('_', ' ').capitalize()} Bootstrap"
                    ),
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.ci_hi,
                    name=f"Lower, n_obs={n_obs}, {bootstrap_method}",
                    legendgroup=f"{bootstrap_method}",
                    legendgrouptitle_text=(
                        f"{bootstrap_method.replace('_', ' ').capitalize()} Bootstrap"
                    ),
                ),
            )

    fig.update_layout(
        title="Means of CI Bounds",
        xaxis_title="late_complier",
        yaxis_title="Mean CI Bounds",
    )

    fig.write_image(path_to_plot_means)
