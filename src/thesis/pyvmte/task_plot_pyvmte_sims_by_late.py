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
idestimands_to_plot = ["late", "sharp"]
constraints_to_plot = [None, "shape_constraints", "mte_monotone", "monotone_response"]


ID_TO_KWARGS = {
    f"{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        constraint=constraint,  # type: ignore[arg-type]
        path_to_plot=BLD
        / "figures"
        / "pyvmte_sims"
        / f"sims_simple_model_by_late_{idestimands}_{constraint}.png",
    )
    for idestimands in idestimands_to_plot
    for constraint in constraints_to_plot
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.plot
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

        bfunc_type_to_color = {"constant": "blue", "bernstein": "red"}

        for bfunc_type in ["constant", "bernstein"]:
            df_plot = df_combined[df_combined["bfunc_type"] == bfunc_type]

            if constraint is not None:
                df_plot = df_plot[df_plot["constraint_type"] == constraint]
                df_plot = df_plot[df_plot["constraint_val"] == _constr_vals[constraint]]
            else:
                df_plot = df_plot[df_plot["constraint_type"] == "none"]

            df_plot = df_plot[df_plot["idestimands"] == idestimands]

            _k_bernstein = df_plot["k_bernstein"].unique()

            assert len(_k_bernstein) == 1

            _legend_title_by_bfunc = {
                "constant": "Constant",
                "bernstein": (
                    f"Bernstein, Degree {int(_k_bernstein[0])}"
                    if bfunc_type == "bernstein"
                    else None
                ),
            }

            fig.add_trace(
                go.Scatter(
                    x=df_plot["b_late"],
                    y=df_plot["covers_true"],
                    mode="lines",
                    name="Coverage of True Parameter",
                    legendgroup=bfunc_type,
                    legendgrouptitle={"text": _legend_title_by_bfunc[bfunc_type]},
                    line={
                        "color": bfunc_type_to_color[bfunc_type],
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

        # Add note with num_gridpoints
        _num_gridpoints = df_plot["num_gridpoints"].unique()
        assert len(_num_gridpoints) == 1
        _num_gridpoints = _num_gridpoints[0]

        fig.add_annotation(
            text=f"Number of gridpoints: {_num_gridpoints}",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.01,
        )

        fig.write_image(path_to_plot)
