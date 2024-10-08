"""Tasks for plotting pyvmte simulation results."""

from pathlib import Path
from typing import Annotated, NamedTuple

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


ID_TO_KWARGS = {
    f"{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        path_to_plot=BLD
        / "figures"
        / "pyvmte_sims"
        / f"sims_simple_model_by_late_{idestimands}_{constraint}.html",
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.wip
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_pyvmte_sims_by_late(
        idestimands: str,
        constraint: str,
        path_to_plot: Annotated[Path, Product],
        path_to_combined: Path = BLD / "data" / "pyvmte_simulations" / "combined.pkl",
    ) -> None:
        """Plot simple model by LATE for different restrictions."""
        df_combined = pd.read_pickle(path_to_combined)

        alpha = 0.05

        # ------------------------------------------------------------------------------
        # Collapse simulation results
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # Plot simulation results
        # ------------------------------------------------------------------------------

        fig = go.Figure()

        confidence_interval_to_color = {"bootstrap": "blue", "subsampling": "red"}
        num_of_obs_to_dash = {1_000: "solid", 10_000: "dash"}

        for confidence_interval in ["bootstrap", "subsampling"]:
            df_plot = df_combined[
                df_combined["confidence_interval"] == confidence_interval
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

            for num_obs in df_plot["num_obs"].unique():
                df_plot_num_obs = df_plot[df_plot["num_obs"] == num_obs]

                fig.add_trace(
                    go.Scatter(
                        x=df_plot_num_obs["late_complier"],
                        y=df_plot_num_obs["covers_true_param"],
                        mode="lines",
                        name=f"N = {num_obs}",
                        legendgroup=confidence_interval,
                        legendgrouptitle={
                            "text": _legend_title_by_confidence_interval[
                                confidence_interval
                            ],
                        },
                        line={
                            "color": confidence_interval_to_color[confidence_interval],
                            "dash": num_of_obs_to_dash[num_obs],
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
            yaxis_title="Bounds",
        )

        # Add note with num_simulations
        num_sims = df_plot["num_sims"].unique()
        assert len(num_sims) == 1
        num_sims = num_sims[0]

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

        fig.write_html(path_to_plot)
