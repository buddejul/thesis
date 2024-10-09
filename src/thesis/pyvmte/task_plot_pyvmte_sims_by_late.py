"""Tasks for plotting pyvmte simulation results."""

from pathlib import Path
from typing import Annotated, NamedTuple
from warnings import warn

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from thesis.config import BLD


class _Arguments(NamedTuple):
    idestimands: str
    constraint: str
    path_to_plot: Annotated[Path, Product]


_shape_constr_to_plot = ("decreasing", "decreasing")
_mte_monotone_to_plot = "decreasing"
_monotone_response_to_plot = "positive"

_constr_vals = {
    "shape_constraints": "_".join(_shape_constr_to_plot),
    "mte_monotone": _mte_monotone_to_plot,
    "monotone_response": _monotone_response_to_plot,
}

_constr_subtitle = {
    "shape_constraints": (
        f"{_constr_vals['shape_constraints'].split('_')[0].capitalize()}"
        " MTR Functions"
    ),
    "mte_monotone": (
        f"{_constr_vals['mte_monotone'].capitalize()} Marginal Treatment Effect"
    ),
    "monotone_response": (
        f"{_constr_vals['monotone_response'].capitalize()} Treatment Response"
    ),
    None: "None",
}

bfunc_types_to_plot = ["constant", "bernstein"]
idestimands_to_plot = ["sharp"]
constraints_to_plot = [None, "mte_monotone", "monotone_response"]

# --------------------------------------------------------------------------------------
# Plot Coverage
# --------------------------------------------------------------------------------------

ID_TO_KWARGS_COVERAGE = {
    f"{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        path_to_plot=BLD
        / "figures"
        / "pyvmte_sims"
        / f"sims_simple_model_by_late_{idestimands}_{constraint}_coverage.html",
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
}

for id_, kwargs in ID_TO_KWARGS_COVERAGE.items():

    @pytask.mark.wip
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_pyvmte_sims_by_late_coverage(  # noqa: PLR0915
        idestimands: str,
        constraint: str,
        path_to_plot: Annotated[Path, Product],
        path_to_sims_combined: Path = BLD
        / "data"
        / "pyvmte_simulations"
        / "combined.pkl",
        path_to_sols_combined: Path = BLD
        / "data"
        / "solutions"
        / "solutions_simple_model_combined.pkl",
        bfunc_type: str = "bernstein",
    ) -> None:
        """Plot simple model by LATE for different restrictions: coverage."""
        df_sims_combined = pd.read_pickle(path_to_sims_combined)
        df_sols_combined = pd.read_pickle(path_to_sols_combined)

        alpha = 0.05

        # ------------------------------------------------------------------------------
        # Plot simulation results
        # ------------------------------------------------------------------------------

        fig = go.Figure()

        # RGB colors with alpha = 0.5
        opacity = 0.25
        confidence_interval_to_color_line = {
            "bootstrap": f"rgba(0, 128, 0, {opacity})",
            "subsampling": f"rgba(0, 0, 255, {opacity})",
        }

        # make markers the same colors but set alpha to 1
        confidence_interval_to_color_marker = {
            key: val[:-4] + "1)"
            for key, val in confidence_interval_to_color_line.items()
        }

        num_of_obs_to_dash = {1_000: "solid", 10_000: "dash"}

        for confidence_interval in ["bootstrap", "subsampling"]:
            df_plot = df_sims_combined[
                df_sims_combined["confidence_interval"] == confidence_interval
            ]

            if constraint is not None:
                df_plot = df_plot[df_plot[constraint] == _constr_vals[constraint]]
            else:
                # Select columns where all three constraints are None
                df_plot = df_plot[
                    (df_plot["shape_constraints"] == "none")
                    & (df_plot["mte_monotone"] == "none")
                    & (df_plot["monotone_response"] == "none")
                ]

            df_plot = df_plot[df_plot["idestimands"] == idestimands]

            _k_bernstein = df_plot["k_bernstein"].unique()
            assert len(_k_bernstein) == 1

            _legend_title_by_confidence_interval = {
                "bootstrap": "Bootstrap",
                "subsampling": "Subsampling",
            }

            # Drop all rows where true_lower_bound and true_upper_bound are NaN
            df_plot = df_plot.dropna(subset=["true_lower_bound", "true_upper_bound"])

            for num_obs in df_plot["num_obs"].unique():
                df_plot_num_obs = df_plot[df_plot["num_obs"] == num_obs]

                fig.add_trace(
                    go.Scatter(
                        x=df_plot_num_obs["late_complier"],
                        y=df_plot_num_obs["covers_true_param"],
                        name=f"N = {num_obs}",
                        legendgroup=confidence_interval,
                        legendgrouptitle={
                            "text": _legend_title_by_confidence_interval[
                                confidence_interval
                            ],
                        },
                        line={
                            "color": confidence_interval_to_color_line[
                                confidence_interval
                            ],
                            "dash": num_of_obs_to_dash[num_obs],
                        },
                        marker={
                            "color": confidence_interval_to_color_marker[
                                confidence_interval
                            ],
                            "size": 10,
                        },
                    ),
                )

        # ------------------------------------------------------------------------------
        # Add true bounds
        # ------------------------------------------------------------------------------

        df_plot_sol = df_sols_combined[df_sols_combined["bfunc_type"] == bfunc_type]

        if constraint is not None:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == constraint]
            df_plot_sol = df_plot_sol[
                df_plot_sol["constraint_val"] == _constr_vals[constraint]
            ]
        else:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == "none"]

        df_plot_sol = df_plot_sol[df_plot_sol["idestimands"] == idestimands]

        _k_bernstein = df_plot_sol["k_bernstein"].unique()

        assert len(_k_bernstein) == 1

        legend_title = "True Bound"

        for bound in ["lower_bound"]:
            fig.add_trace(
                go.Scatter(
                    x=df_plot_sol["b_late"],
                    y=df_plot_sol[bound],
                    mode="lines",
                    name=f"{bound.split('_')[0].capitalize()} Bound",
                    legendgroup=bfunc_type,
                    legendgrouptitle={"text": legend_title},
                    line={
                        "color": "rgba(255, 0, 0, 0.2)",
                    },
                    yaxis="y2",
                ),
            )

        _subtitle = (
            f" <br><sup> Identified Estimands: {idestimands.capitalize()},"
            f" alpha = {alpha} </sup>"
            f" <br><sup> Shape constraints: {_constr_subtitle[constraint]} </sup>"
        )

        fig.update_layout(
            title="Coverage for Target LATE(0.4, 0.8) for Binary-IV Model" + _subtitle,
            xaxis_title="Identified LATE",
            yaxis_title="Mean Coverage",
            yaxis2={"title": "", "overlaying": "y", "side": "right"},
        )

        # Add note with num_simulations
        num_sims = df_plot["num_sims"].unique()

        if len(num_sims) != 1:
            # Get counts of num_sims
            counts = df_plot["num_sims"].value_counts()
            warning = f"num_sims is not unique, got {counts}."
            warn(warning, stacklevel=2)
        num_sims = np.max(num_sims)

        num_boot = df_plot["n_boot"].unique()
        assert len(num_boot) == 1
        num_boot = num_boot[0]

        num_subsamples = df_plot["n_subsamples"].unique()
        assert len(num_subsamples) == 1
        num_subsamples = num_subsamples[0]

        if confidence_interval == "bootstrap":
            subsample_size = None
        else:
            subsample_size = df_plot["subsample_size"].unique()

        fig.add_annotation(
            text=(
                f"N Simulations: {int(num_sims)}<br>"
                f"Subsample size: {subsample_size}<br>"
                f"N Bootstrap Samples/Subsamples: {num_boot}/{num_subsamples}"
            ),
            font={"size": 10},
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1,
            y=-0.21,
            # Right aligned
            align="right",
        )

        # Draw horizontal line at 1 - alpha
        fig.add_shape(
            type="line",
            x0=0,
            y0=1 - alpha,
            x1=1,
            y1=1 - alpha,
            line={"color": "rgba(0, 0, 0, 0.2)", "width": 2},
        )

        fig.write_html(path_to_plot)


# --------------------------------------------------------------------------------------
# Plot Means
# --------------------------------------------------------------------------------------

ID_TO_KWARGS_MEANS = {
    f"{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        path_to_plot=BLD
        / "figures"
        / "pyvmte_sims"
        / f"sims_simple_model_by_late_{idestimands}_{constraint}_means.html",
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
}

for id_, kwargs in ID_TO_KWARGS_MEANS.items():

    @pytask.mark.wip
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_pyvmte_sims_by_late_means(
        idestimands: str,
        constraint: str,
        path_to_plot: Annotated[Path, Product],
        path_to_sims_combined: Path = BLD
        / "data"
        / "pyvmte_simulations"
        / "combined.pkl",
        path_to_solutions_combined: Path = BLD
        / "data"
        / "solutions"
        / "solutions_simple_model_combined.pkl",
        bfunc_type="bernstein",
    ) -> None:
        """Plot simple model by LATE for different restrictions: means."""
        df_sims_combined = pd.read_pickle(path_to_sims_combined)
        df_sols_combined = pd.read_pickle(path_to_solutions_combined)

        alpha = 0.05

        # ------------------------------------------------------------------------------
        # Plot simulation results
        # ------------------------------------------------------------------------------

        cols_to_plot = [
            "sim_ci_lower",
            "sim_ci_upper",
            "sim_lower_bound",
            "sim_upper_bound",
            # "true_lower_bound",
            # "true_upper_bound",
        ]

        col_to_legend_group = {
            "sim_ci_lower": "Confidence Interval",
            "sim_ci_upper": "Confidence Interval",
            "sim_lower_bound": "Estimated",
            "sim_upper_bound": "Estimated",
            "true_lower_bound": "True",
            "true_upper_bound": "True",
        }

        col_to_show_legend = {
            "sim_ci_lower": True,
            "sim_ci_upper": False,
            "sim_lower_bound": True,
            "sim_upper_bound": False,
            "true_lower_bound": True,
            "true_upper_bound": False,
        }

        # RGB colors with alpha = 0.5
        opacity = 0.25
        col_to_color_line = {
            "sim_ci_lower": f"rgba(0, 128, 0, {opacity})",
            "sim_ci_upper": f"rgba(0, 128, 0, {opacity})",
            "sim_lower_bound": f"rgba(0, 0, 255, {opacity})",
            "sim_upper_bound": f"rgba(0, 0, 255, {opacity})",
            "true_lower_bound": f"rgba(255, 0, 0, {opacity})",
            "true_upper_bound": f"rgba(255, 0, 0, {opacity})",
        }

        # make markers the same colors but set alpha to 1
        col_to_color_marker = {
            key: val[:-4] + "1)" for key, val in col_to_color_line.items()
        }

        marker_symbol_to_num_obs = {
            1_000: "circle",
            10_000: "diamond",
        }

        fig = go.Figure()

        num_of_obs_to_dash = {1_000: "solid", 10_000: "dash"}

        df_plot = df_sims_combined[
            df_sims_combined["confidence_interval"] == "bootstrap"
        ]

        if constraint is not None:
            df_plot = df_plot[df_plot[constraint] == _constr_vals[constraint]]
        else:
            # Select columns where all three constraints are None
            df_plot = df_plot[
                (df_plot["shape_constraints"] == "none")
                & (df_plot["mte_monotone"] == "none")
                & (df_plot["monotone_response"] == "none")
            ]

        df_plot = df_plot[df_plot["idestimands"] == idestimands]

        _k_bernstein = df_plot["k_bernstein"].unique()
        assert len(_k_bernstein) == 1

        # Drop all rows where true_lower_bound and true_upper_bound are NaN
        df_plot = df_plot.dropna(subset=["true_lower_bound", "true_upper_bound"])

        for col in cols_to_plot:
            for num_obs in df_plot["num_obs"].unique():
                df_plot_num_obs = df_plot[df_plot["num_obs"] == num_obs]

                fig.add_trace(
                    go.Scatter(
                        x=df_plot_num_obs["late_complier"],
                        y=df_plot_num_obs[col],
                        name=f"N = {num_obs}",
                        legendgroup=col_to_legend_group[col],
                        legendgrouptitle={
                            "text": col_to_legend_group[col],
                        },
                        marker={
                            "color": col_to_color_marker[col],
                            "symbol": marker_symbol_to_num_obs[num_obs],
                            "size": 10,
                        },
                        line={
                            "color": col_to_color_line[col],
                            "dash": num_of_obs_to_dash[num_obs],
                        },
                        showlegend=col_to_show_legend[col],
                    ),
                )

        # ------------------------------------------------------------------------------
        # Add true bounds
        # ------------------------------------------------------------------------------
        bound_to_dash = {"upper_bound": "solid", "lower_bound": "solid"}

        df_plot_sol = df_sols_combined[df_sols_combined["bfunc_type"] == bfunc_type]

        if constraint is not None:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == constraint]
            df_plot_sol = df_plot_sol[
                df_plot_sol["constraint_val"] == _constr_vals[constraint]
            ]
        else:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == "none"]

        df_plot_sol = df_plot_sol[df_plot_sol["idestimands"] == idestimands]

        _k_bernstein = df_plot_sol["k_bernstein"].unique()

        assert len(_k_bernstein) == 1

        legend_title = "True Bounds"

        for bound in ["upper_bound", "lower_bound"]:
            fig.add_trace(
                go.Scatter(
                    x=df_plot_sol["b_late"],
                    y=df_plot_sol[bound],
                    mode="lines",
                    name=f"{bound.split('_')[0].capitalize()} Bound",
                    legendgroup=bfunc_type,
                    legendgrouptitle={"text": legend_title},
                    line={
                        "color": "red",
                        "dash": bound_to_dash[bound],
                    },
                ),
            )

        _subtitle = (
            f" <br><sup> Identified Estimands: {idestimands.capitalize()},"
            f" alpha = {alpha} </sup>"
            f" <br><sup> Shape constraints: {_constr_subtitle[constraint]} </sup>"
        )

        fig.update_layout(
            title="Coverage for Target LATE(0.4, 0.8) for Binary-IV Model" + _subtitle,
            xaxis_title="Identified LATE",
            yaxis_title="Means",
        )

        # Add note with num_simulations
        num_sims = df_plot["num_sims"].unique()

        if len(num_sims) != 1:
            # Get counts of num_sims
            counts = df_plot["num_sims"].value_counts()
            warning = f"num_sims is not unique, got {counts}."
            warn(warning, stacklevel=2)
        num_sims = np.max(num_sims)

        fig.add_annotation(
            text=(f"N Simulations: {int(num_sims)}<br>"),
            font={"size": 10},
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1,
            y=-0.21,
            # Right aligned
            align="right",
        )

        # Make x-axis from 0 to 1
        fig.update_xaxes(range=[0, 1])

        fig.write_html(path_to_plot)
