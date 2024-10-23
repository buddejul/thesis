"""Tasks for plotting pyvmte simulation results."""

from pathlib import Path
from typing import Annotated, NamedTuple
from warnings import warn

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from plotly.io import write_image  # type: ignore[import-untyped]
from pytask import Product, task

from thesis.config import BLD
from thesis.pyvmte._task_pyvmte_sims import grid_by_constraint


class _Arguments(NamedTuple):
    idestimands: str
    constraint: str
    problematic_region: np.ndarray
    path_to_plot: Annotated[Path, Product]
    path_to_plot_problematic_region: Annotated[Path, Product]
    path_to_plot_html: Annotated[Path, Product] | None = None
    path_to_plot_problematic_region_html: Annotated[Path, Product] | None = None
    confidence_interval: str | None = None


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
    "none": "None",
}

bfunc_types_to_plot = ["constant", "bernstein"]
idestimands_to_plot = ["sharp"]
constraints_to_plot = ["none", "mte_monotone", "monotone_response"]

# --------------------------------------------------------------------------------------
# Plot Coverage
# --------------------------------------------------------------------------------------

plot_dir = BLD / "figures" / "pyvmte_sims"

ID_TO_KWARGS_COVERAGE = {
    f"{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        problematic_region=grid_by_constraint[constraint],
        path_to_plot=plot_dir
        / f"sims_binary_iv_{idestimands}_{constraint}_coverage.png",
        path_to_plot_problematic_region=plot_dir
        / (
            f"sims_binary_iv_{idestimands}_{constraint}"
            "_coverage_problematic_region.png"
        ),
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
}

for id_, kwargs in ID_TO_KWARGS_COVERAGE.items():
    ID_TO_KWARGS_COVERAGE[id_] = kwargs._replace(
        path_to_plot_html=ID_TO_KWARGS_COVERAGE[id_].path_to_plot.with_suffix(".html"),
        path_to_plot_problematic_region_html=ID_TO_KWARGS_COVERAGE[
            id_
        ].path_to_plot_problematic_region.with_suffix(".html"),
    )

for id_, kwargs in ID_TO_KWARGS_COVERAGE.items():

    @pytask.mark.wip
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_pyvmte_sims_by_late_coverage(  # noqa: PLR0915
        idestimands: str,
        constraint: str,
        problematic_region: np.ndarray,
        path_to_plot: Annotated[Path, Product],
        path_to_plot_problematic_region: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        path_to_plot_problematic_region_html: Annotated[Path, Product],
        path_to_sims_combined: Path = BLD
        / "data"
        / "pyvmte_simulations"
        / "combined.pkl",
        path_to_sols_combined: Path = BLD
        / "data"
        / "solutions"
        / "solutions_simple_model_combined.pkl",
        bfunc_type: str = "bernstein",
        confidence_interval: str | None = None,
    ) -> None:
        """Plot simple model by LATE for different restrictions: coverage."""
        del confidence_interval

        df_sims_combined = pd.read_pickle(path_to_sims_combined)
        df_sols_combined = pd.read_pickle(path_to_sols_combined)

        alpha = 0.05
        1 - alpha

        # ------------------------------------------------------------------------------
        # Plot simulation results
        # ------------------------------------------------------------------------------

        fig = go.Figure()

        # RGB colors with alpha = 0.5
        opacity = 1
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

            if constraint != "none":
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
            _k_bernstein = _k_bernstein[0]

            _legend_title_by_confidence_interval = {
                "bootstrap": "Bootstrap",
                "subsampling": "Subsampling",
            }

            # Drop all rows where true_lower_bound and true_upper_bound are NaN
            df_plot = df_plot.dropna(subset=["true_lower_bound", "true_upper_bound"])

            for num_obs in df_plot["num_obs"].unique():
                df_plot_num_obs = df_plot[df_plot["num_obs"] == num_obs]
                df_plot_num_obs = df_plot_num_obs.sort_values("late_complier")

                fig.add_trace(
                    go.Scatter(
                        x=df_plot_num_obs["late_complier"],
                        y=df_plot_num_obs["covers_true_param"],
                        mode="markers+lines",
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
                            "size": 5,
                            "color": confidence_interval_to_color_marker[
                                confidence_interval
                            ],
                        },
                    ),
                )

        # ------------------------------------------------------------------------------
        # Add true bounds
        # ------------------------------------------------------------------------------

        df_plot_sol = df_sols_combined[df_sols_combined["bfunc_type"] == bfunc_type]

        if constraint != "none":
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == constraint]
            df_plot_sol = df_plot_sol[
                df_plot_sol["constraint_val"] == _constr_vals[constraint]
            ]
        else:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == "none"]

        df_plot_sol = df_plot_sol[df_plot_sol["idestimands"] == idestimands]

        df_plot_sol = df_plot_sol[df_plot_sol["k_bernstein"] == _k_bernstein]

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
            f" Nominal Coverage = {1- alpha} </sup>"
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
                f"N Simulations: {num_sims}<br>"
                f"Subsample size: {subsample_size}<br>"
                f"N Bootstrap Samples/Subsamples: {int(num_boot)}/{int(num_subsamples)}"
            ),
            font={"size": 10},
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.2,
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

        # Make x-axis from 0 to 1
        fig.update_xaxes(range=[-0.1, 1])

        write_image(fig, path_to_plot, scale=2)
        fig.write_html(path_to_plot_html)

        # Restrict x-axis to problematic region
        fig.update_xaxes(range=[np.min(problematic_region), np.max(problematic_region)])

        write_image(fig, path_to_plot_problematic_region, scale=2)
        fig.write_html(path_to_plot_problematic_region_html)


# --------------------------------------------------------------------------------------
# Plot Means
# --------------------------------------------------------------------------------------

ID_TO_KWARGS_MEANS = {
    f"{idestimands}_{constraint}_{confidence_interval}": _Arguments(
        confidence_interval=confidence_interval,
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        problematic_region=grid_by_constraint[constraint],
        path_to_plot=plot_dir
        / (
            f"sims_binary_iv_{idestimands}_{constraint}_means_"
            f"{confidence_interval}.png"
        ),
        path_to_plot_problematic_region=plot_dir
        / (
            f"sims_binary_iv_{idestimands}_{constraint}_means_"
            f"problematic_region_{confidence_interval}.png"
        ),
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
    for confidence_interval in ["bootstrap", "subsampling"]
}

for id_, kwargs in ID_TO_KWARGS_MEANS.items():
    ID_TO_KWARGS_MEANS[id_] = kwargs._replace(
        path_to_plot_html=ID_TO_KWARGS_MEANS[id_].path_to_plot.with_suffix(".html"),
        path_to_plot_problematic_region_html=ID_TO_KWARGS_MEANS[
            id_
        ].path_to_plot_problematic_region.with_suffix(".html"),
    )

for id_, kwargs in ID_TO_KWARGS_MEANS.items():

    @pytask.mark.wip
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_pyvmte_sims_by_late_means(  # noqa: PLR0915
        confidence_interval: str,
        idestimands: str,
        constraint: str,
        problematic_region: np.ndarray,
        path_to_plot: Annotated[Path, Product],
        path_to_plot_problematic_region: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        path_to_plot_problematic_region_html: Annotated[Path, Product],
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
            df_sims_combined["confidence_interval"] == confidence_interval
        ]

        if constraint != "none":
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
        _k_bernstein = _k_bernstein[0]

        # Drop all rows where true_lower_bound and true_upper_bound are NaN
        df_plot = df_plot.dropna(subset=["true_lower_bound", "true_upper_bound"])

        for col in cols_to_plot:
            for num_obs in df_plot["num_obs"].unique():
                df_plot_num_obs = df_plot[df_plot["num_obs"] == num_obs]

                df_plot_num_obs = df_plot_num_obs.sort_values("late_complier")

                fig.add_trace(
                    go.Scatter(
                        x=df_plot_num_obs["late_complier"],
                        y=df_plot_num_obs[col],
                        mode="markers+lines",
                        name=f"N = {num_obs}",
                        legendgroup=col_to_legend_group[col],
                        legendgrouptitle={
                            "text": col_to_legend_group[col],
                        },
                        marker={
                            "size": 5,
                            "color": col_to_color_marker[col],
                            "symbol": marker_symbol_to_num_obs[num_obs],
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

        if constraint != "none":
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == constraint]
            df_plot_sol = df_plot_sol[
                df_plot_sol["constraint_val"] == _constr_vals[constraint]
            ]
        else:
            df_plot_sol = df_plot_sol[df_plot_sol["constraint_type"] == "none"]

        df_plot_sol = df_plot_sol[df_plot_sol["idestimands"] == idestimands]
        df_plot_sol = df_plot_sol[df_plot_sol["k_bernstein"] == _k_bernstein]

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
                        "color": "rgba(255, 0, 0, 0.5)",
                        "dash": bound_to_dash[bound],
                    },
                ),
            )

        _subtitle = (
            f"<br><sup>Confidence Interval: {confidence_interval.capitalize()}, "
            f"Identified: {idestimands.capitalize()}"
            f"<br>Nominal Coverage = {1 - alpha}, "
            f"Shape constraints: {_constr_subtitle[constraint]} </sup>"
        )

        fig.update_layout(
            title="Mean Estimates for Target LATE(0.4, 0.8) for Binary-IV Model"
            + _subtitle,
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
            x=1.2,
            y=-0.21,
            # Right aligned
            align="right",
        )

        # Make x-axis from 0 to 1
        fig.update_xaxes(range=[-0.1, 1])

        write_image(fig, path_to_plot, scale=2)
        fig.write_html(path_to_plot_html)

        fig.update_xaxes(range=[np.min(problematic_region), np.max(problematic_region)])

        write_image(fig, path_to_plot_problematic_region, scale=2)
        fig.write_html(path_to_plot_problematic_region_html)
