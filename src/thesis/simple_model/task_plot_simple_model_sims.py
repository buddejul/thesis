"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.simple_model.task_simple_model_sims import (
    EPS_FUNS_NUMERICAL_DELTA,
)
from thesis.utilities import get_func_as_string

EPS_FUNS_STRINGS = [get_func_as_string(eps_fun) for eps_fun in EPS_FUNS_NUMERICAL_DELTA]
KAPPA_FUNS_STRINGS = [
    get_func_as_string(kappa_fun) for kappa_fun in EPS_FUNS_NUMERICAL_DELTA
]

<<<<<<< HEAD
KAPPA_TO_PLOT = "np.sqrt(np.log(n))"

=======
>>>>>>> main

# TODO(@buddejul): Put graphs below in a loop, currently there is a lot of copy/paste.
# TODO(@buddejul): Include true parameters in plots.
# TODO(@buddejul): Split tasks, function is too complex, see noqa below.
<<<<<<< HEAD
def task_plot_simple_model_sims(  # noqa: C901, PLR0912, PLR0915
=======
def task_plot_simple_model_sims(  # noqa: C901, PLR0912
>>>>>>> main
    path_to_data: Path = BLD / "simple_model" / "sim_results_combined.pkl",
    path_to_plot_coverage: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "coverage.png",
    ),
    path_to_plot_length: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "length.png",
    ),
    path_to_plot_means_hi: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "means_hi.png",
    ),
    path_to_plot_means_lo: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "means_lo.png",
    ),
    path_to_plot_coverage_by_eps_fun: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "coverage_eps_fun.png",
    ),
    path_to_plot_coverage_by_kappa_fun: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "figures" / "coverage_kappa_fun.png",
    ),
) -> None:
    """Plot the coverage probability of the confidence interval."""
    data = pd.read_pickle(path_to_data)  # noqa: S301

    # TODO(@buddejul): Do this properly and loop over mtr constraints.
    data = data[data.constraint_mtr == "increasing"]

    data = data.sort_values("late_complier")

<<<<<<< HEAD
    # Check if n_boot and n_sims are unique; if so store else error.
    if len(data.n_boot.unique()) > 1:
        msg = "n_boot is not unique."
        raise ValueError(msg)
    n_sims = data.n_sims.unique()[0]
    if len(data.n_sims.unique()) > 1:
        msg = "n_sims is not unique."
        raise ValueError(msg)
    n_boot = data.n_boot.unique()[0]

    subtitle = (
        f"<br><sup>Simulations = {n_sims}, Bootstrap Repetitions = {n_boot}</sup>"
    )

=======
>>>>>>> main
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
<<<<<<< HEAD
        "np.sqrt(np.log(n))": "green",
=======
        "np.log(n)pow(1div2)": "green",
>>>>>>> main
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
<<<<<<< HEAD
                    data_sub = data_sub[data_sub["kappa_fun"] == KAPPA_TO_PLOT]
=======
                    data_sub = data_sub[data_sub["kappa_fun"] == "np.log(n)pow(1div2)"]
>>>>>>> main

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
<<<<<<< HEAD
            title=(f"{col_to_plot.capitalize()} of CI for true parameter" + subtitle),
=======
            title=f"{col_to_plot.capitalize()} of CI for true parameter",
>>>>>>> main
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
    params_to_stat = {
<<<<<<< HEAD
        "lo": {"title": "Means of CI Lower Bounds", "path": path_to_plot_means_lo},
        "hi": {"title": "Means of CI Upper Bounds", "path": path_to_plot_means_hi},
    }

    for stat in ["hi", "lo"]:
=======
        "ci_lo": {"title": "Means of CI Lower Bounds", "path": path_to_plot_means_lo},
        "ci_hi": {"title": "Means of CI Upper Bounds", "path": path_to_plot_means_hi},
    }

    for stat in ["ci_hi", "ci_lo"]:
>>>>>>> main
        fig = go.Figure()
        for n_obs in data.n_obs.unique():
            for bootstrap_method in ["standard", "numerical_delta", "analytical_delta"]:
                data_sub = data[
                    (data.n_obs == n_obs) & (data.bootstrap_method == bootstrap_method)
                ]
                if bootstrap_method == "numerical_delta":
                    data_sub = data_sub[data_sub["eps_fun"] == "npow(-1div2)"]
                if bootstrap_method == "analytical_delta":
<<<<<<< HEAD
                    data_sub = data_sub[data_sub["kappa_fun"] == KAPPA_TO_PLOT]
=======
                    data_sub = data_sub[data_sub["kappa_fun"] == "npow(1div2)"]
>>>>>>> main
                fig.add_trace(
                    go.Scatter(
                        x=data_sub.late_complier,
                        y=data_sub[stat],
                        name=f"n_obs={n_obs}",
                        legendgroup=f"{bootstrap_method}",
                        legendgrouptitle_text=(
                            f"{bootstrap_method.replace('_', ' ').capitalize()}"
                            "Bootstrap"
                        ),
                        line={
                            "color": color_by_bootstrap_method[bootstrap_method],
                            "dash": line_type_by_n_obs[int(n_obs)],
                        },
                    ),
                )

        fig.update_layout(
<<<<<<< HEAD
            title=(params_to_stat[stat]["title"] + subtitle),  # type: ignore[operator]
=======
            title=params_to_stat[stat]["title"],
>>>>>>> main
            xaxis_title="late_complier",
            yaxis_title="Mean CI Bounds",
        )

        fig.write_image(params_to_stat[stat]["path"])

    # ==================================================================================
    # Plot coverage by eps_fun
    # ==================================================================================
    fig = go.Figure()
<<<<<<< HEAD
=======

>>>>>>> main
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
<<<<<<< HEAD
            title="Coverage by eps_fun" + subtitle,
=======
            title="Coverage by eps_fun",
>>>>>>> main
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
<<<<<<< HEAD
            title="Coverage by kappa_fun" + subtitle,
=======
            title="Coverage by kappa_fun",
>>>>>>> main
            xaxis_title="late_complier",
            yaxis_title="Coverage",
        )

    fig.write_image(path_to_plot_coverage_by_kappa_fun)
