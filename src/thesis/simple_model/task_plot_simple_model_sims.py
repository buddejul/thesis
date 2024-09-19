"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.simple_model.task_simple_model_sims import (
    EPS_FUNS_NUMERICAL_DELTA,
    ID_TO_KWARGS,
    _Arguments,
)
from thesis.utilities import get_func_as_string

EPS_FUNS_STRINGS = [get_func_as_string(eps_fun) for eps_fun in EPS_FUNS_NUMERICAL_DELTA]
KAPPA_FUNS_STRINGS = [
    get_func_as_string(kappa_fun) for kappa_fun in EPS_FUNS_NUMERICAL_DELTA
]


# TODO(@buddejul): Put graphs below in a loop, currently there is a lot of copy/paste.
# TODO(@buddejul): Include true parameters in plots.
# TODO(@buddejul): Split tasks, function is too complex, see noqa below.
def task_plot_bootstrap_sims(  # noqa: C901, PLR0912, PLR0915
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
    path_to_plot_coverage_by_eps_fun: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "coverage_eps_fun.png",
    ),
    path_to_plot_coverage_by_kappa_fun: Annotated[Path, Product] = Path(
        BLD / "boot" / "figures" / "coverage_kappa_fun.png",
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
                get_func_as_string(kwargs.bootstrap_params["eps_fun"]),
                get_func_as_string(kwargs.bootstrap_params["kappa_fun"]),
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
        "eps_fun",
        "kappa_fun",
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
        "analytical_delta": "green",
    }

    color_by_eps_fun = {
        "npow(-1div1)": "black",
        "npow(-1div2)": "blue",
        "npow(-1div3)": "red",
        "npow(-1div4)": "green",
        "npow(-1div5)": "orange",
        "npow(-1div6)": "purple",
    }

    color_by_kappa_fun = {
        "npow(1div2)": "blue",
        "npow(1div3)": "red",
        "npow(1div6)": "purple",
        "np.log(n)pow(1div2)": "green",
        "(2xnp.log(np.log(n)))pow(1div2)": "orange",
    }

    line_type_by_n_obs = {
        250: "solid",
        1_000: "dash",
        10_000: "dot",
    }

    for col_to_plot in ["coverage", "length"]:
        fig = go.Figure()
        for n_obs in data.n_obs.unique():
            for bootstrap_method in ["standard", "numerical_delta", "analytical_delta"]:
                data_sub = data[
                    (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
                ]
                if bootstrap_method == "numerical_delta":
                    data_sub = data_sub[data_sub["eps_fun"] == "npow(-1div2)"]
                if bootstrap_method == "analytical_delta":
                    data_sub = data_sub[data_sub["kappa_fun"] == "np.log(n)pow(1div2)"]

                fig.add_trace(
                    go.Scatter(
                        x=data_sub.late_complier,
                        y=data_sub[col_to_plot],
                        name=f"n_obs={n_obs}",
                        legendgroup=f"{bootstrap_method}",
                        legendgrouptitle_text=(
                            f"{bootstrap_method.replace('_', ' ').capitalize()} "
                            "Bootstrap"
                        ),
                        line={
                            "color": color_by_bootstrap_method[bootstrap_method],
                            "dash": line_type_by_n_obs[int(n_obs)],
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
        for bootstrap_method in ["standard", "numerical_delta", "analytical_delta"]:
            data_sub = data[
                (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
            ]
            if bootstrap_method == "numerical_delta":
                data_sub = data_sub[data_sub["eps_fun"] == "npow(-1div2)"]
            if bootstrap_method == "analytical_delta":
                data_sub = data_sub[data_sub["kappa_fun"] == "npow(1div2)"]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.ci_lo,
                    name=f"n_obs={n_obs}",
                    legendgroup=f"{bootstrap_method}",
                    legendgrouptitle_text=(
                        f"{bootstrap_method.replace('_', ' ').capitalize()} Bootstrap"
                    ),
                    line={
                        "color": color_by_bootstrap_method[bootstrap_method],
                        "dash": line_type_by_n_obs[int(n_obs)],
                    },
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.ci_hi,
                    name=f"Lower, N ={n_obs}, {bootstrap_method}",
                    legendgroup=f"{bootstrap_method}",
                    legendgrouptitle_text=(
                        f"{bootstrap_method.replace('_', ' ').capitalize()} Bootstrap"
                    ),
                    line={
                        "color": color_by_bootstrap_method[bootstrap_method],
                        "dash": line_type_by_n_obs[int(n_obs)],
                    },
                    showlegend=False,
                ),
            )

    fig.update_layout(
        title="Means of CI Bounds",
        xaxis_title="late_complier",
        yaxis_title="Mean CI Bounds",
    )

    fig.write_image(path_to_plot_means)

    # ==================================================================================
    # Plot coverage by eps_fun
    # ==================================================================================
    fig = go.Figure()

    for eps_fun in data.eps_fun.unique():
        eps_fun_to_print = (
            eps_fun.replace("npow", "n^")
            .replace("div", "/")
            .replace("(", "")
            .replace(")", "")
        )

        for n_obs in data.n_obs.unique():
            data_sub = data[data.eps_fun == eps_fun]
            data_sub = data_sub[data_sub.n_obs == n_obs]
            data_sub = data_sub[data_sub["bootstrap_method"] == "numerical_delta"]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.coverage,
                    name=f"n_obs={n_obs}",
                    legendgroup=f"{eps_fun}",
                    legendgrouptitle_text=(f"eps(n) = {eps_fun_to_print}"),
                    line={
                        "dash": line_type_by_n_obs[int(n_obs)],
                        "width": 2,
                        "color": color_by_eps_fun[eps_fun],
                    },
                ),
            )

        fig.update_layout(
            title="Coverage by eps_fun",
            xaxis_title="late_complier",
            yaxis_title="Coverage",
        )

    fig.write_image(path_to_plot_coverage_by_eps_fun)

    # ==================================================================================
    # Plot coverage by kappa_fun
    # ==================================================================================
    fig = go.Figure()

    for kappa_fun in data.kappa_fun.unique():
        kappa_fun_to_print = (
            kappa_fun.replace("npow", "n^")
            .replace("div", "/")
            .replace("(", "")
            .replace(")", "")
        )

        for n_obs in data.n_obs.unique():
            data_sub = data[data.kappa_fun == kappa_fun]
            data_sub = data_sub[data_sub.n_obs == n_obs]
            data_sub = data_sub[data_sub["bootstrap_method"] == "analytical_delta"]
            fig.add_trace(
                go.Scatter(
                    x=data_sub.late_complier,
                    y=data_sub.coverage,
                    name=f"n_obs={n_obs}",
                    legendgroup=f"{kappa_fun}",
                    legendgrouptitle_text=(f"eps(n) = {kappa_fun_to_print}"),
                    line={
                        "dash": line_type_by_n_obs[int(n_obs)],
                        "width": 2,
                        "color": color_by_kappa_fun[kappa_fun],
                    },
                ),
            )

        fig.update_layout(
            title="Coverage by kappa_fun",
            xaxis_title="late_complier",
            yaxis_title="Coverage",
        )

    fig.write_image(path_to_plot_coverage_by_kappa_fun)
