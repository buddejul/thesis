"""Tasks for plotting pyvmte simulation results: different alpha for CI."""

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
from thesis.pyvmte.sims.task_pyvmte_sims import grid_by_constraint


class _Arguments(NamedTuple):
    idestimands: str
    constraint: str
    problematic_region: np.ndarray
    path_to_plot: Annotated[Path, Product]
    path_to_plot_problematic_region: Annotated[Path, Product]
    lp_tolerance: str
    path_to_plot_html: Annotated[Path, Product] | None = None
    path_to_plot_problematic_region_html: Annotated[Path, Product] | None = None
    confidence_interval: str = "subsampling"


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

confidence_intervals_to_plot = ["bootstrap", "subsampling"]
idestimands_to_plot = ["sharp"]
constraints_to_plot = ["none"]
tolerances_to_plot = ["1/n"]
subsample_share_to_plot = 0.1
num_obs_to_plot = [10_000]
alpha_im_crit_to_plot = [False, True]

# --------------------------------------------------------------------------------------
# Plotting Settings
# --------------------------------------------------------------------------------------

alpha_im_crit_to_dash = {False: "solid", True: "dash"}
alpha_im_crit_to_marker = {False: "circle", True: "diamond"}

confidence_interval_to_color = {
    "bootstrap": "rgba(0, 128, 0, 1)",
    "subsampling": "rgba(0, 0, 255, 1)",
}

name_by_alpha_im_crit = {
    False: "Alpha Critival Value",
    True: "IM Critival Value",
}

# --------------------------------------------------------------------------------------
# Plot Coverage
# --------------------------------------------------------------------------------------

plot_dir = BLD / "figures" / "pyvmte_sims"

ID_TO_KWARGS_COVERAGE = {
    f"{idestimands}_{constraint}_{lp_tolerance}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        problematic_region=grid_by_constraint[constraint],
        path_to_plot=plot_dir
        / "plot_by_im_alpha_crit"
        / f"sims_binary_iv_{idestimands}_{constraint}_coverage_by_alpha.png",
        path_to_plot_problematic_region=plot_dir
        / "plot_by_im_alpha_crit"
        / (
            f"sims_binary_iv_{idestimands}_{constraint}"
            f"_coverage_problematic_region_by_alpha.png"
        ),
        lp_tolerance=lp_tolerance,
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
    for lp_tolerance in tolerances_to_plot
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
    def task_plot_pyvmte_sims_by_late_and_alpha_coverage(  # noqa: PLR0915
        idestimands: str,
        constraint: str,
        problematic_region: np.ndarray,
        path_to_plot: Annotated[Path, Product],
        path_to_plot_problematic_region: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        path_to_plot_problematic_region_html: Annotated[Path, Product],
        lp_tolerance: str,
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

        # ------------------------------------------------------------------------------
        # Plot simulation results
        # ------------------------------------------------------------------------------

        fig = go.Figure()

        for confidence_interval in confidence_intervals_to_plot:
            idx = df_sims_combined["lp_tolerance"] == lp_tolerance
            df_plot = df_sims_combined[idx]

            df_plot = df_plot[df_plot["confidence_interval"] == confidence_interval]

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

            if confidence_interval == "subsampling":
                df_plot = df_plot[df_plot["subsample_share"] == subsample_share_to_plot]

            _k_bernstein = df_plot["k_bernstein"].unique()
            assert len(_k_bernstein) == 1
            _k_bernstein = _k_bernstein[0]

            _legend_title_by_confidence_interval = {
                "subsampling": f"Subsampling (N = {num_obs_to_plot[0]})",
                "bootstrap": f"Bootstrap (N = {num_obs_to_plot[0]})",
            }

            # Drop all rows where true_lower_bound and true_upper_bound are NaN
            df_plot = df_plot.dropna(subset=["true_lower_bound", "true_upper_bound"])

            for num_obs in num_obs_to_plot:
                for alpha_im_crit in alpha_im_crit_to_plot:
                    idx = df_plot["alpha_im_crit"] == alpha_im_crit
                    _df = df_plot[idx]

                    _df = _df[_df["num_obs"] == num_obs]
                    _df = _df.sort_values("late_complier")

                    assert _df.set_index("late_complier").index.is_unique

                    fig.add_trace(
                        go.Scatter(
                            x=_df["late_complier"],
                            y=_df["covers_true_param"],
                            mode="markers+lines",
                            name=f"{name_by_alpha_im_crit[alpha_im_crit]}",
                            legendgroup=confidence_interval,
                            legendgrouptitle={
                                "text": _legend_title_by_confidence_interval[
                                    confidence_interval
                                ],
                            },
                            line={
                                "color": confidence_interval_to_color[
                                    confidence_interval
                                ],
                                "dash": alpha_im_crit_to_dash[alpha_im_crit],
                            },
                            marker={
                                "size": 6,
                                "color": confidence_interval_to_color[
                                    confidence_interval
                                ],
                                "symbol": alpha_im_crit_to_marker[alpha_im_crit],
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

        num_subsamples = df_plot["n_subsamples"].unique()
        assert len(num_subsamples) == 1
        num_subsamples = num_subsamples[0]

        fig.add_annotation(
            text=(f"N Subsamples: {int(num_subsamples)}"),
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
# Plot Average
