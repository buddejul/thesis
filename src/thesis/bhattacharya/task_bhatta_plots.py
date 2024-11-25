"""Tasks for plotting Bhattachayra 2009 simulations."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from thesis.bhattacharya.task_bhatta_sims import ID_TO_KWARGS
from thesis.config import BLD

paths_to_sim_res = [kwargs.path_to_res for kwargs in ID_TO_KWARGS.values()]


class _ArgsPlots(NamedTuple):
    num_obs: int
    stat_to_plot: str
    path_to_plot: Path


ID_TO_KWARGS_PLOTS = {
    f"{num_obs}_{stat_to_plot}": _ArgsPlots(
        num_obs=num_obs,
        stat_to_plot=stat_to_plot,
        path_to_plot=BLD / "bhatta" / "plots" / f"{stat_to_plot}_{num_obs}.png",
    )
    for num_obs in [100, 1_000, 10_000]
    for stat_to_plot in [
        "covers",
        "covers_lo",
        "covers_hi",
        "v_hat",
        "num_solutions",
        "ci_lo",
        "ci_hi",
        "ci_midpoint",
        "z_lo",
        "z_hi",
    ]
}


@pytask.mark.bhatta()
def task_combine_bhatta_sims(
    paths_to_sim_res: list[Path] = paths_to_sim_res,
    path_to_combined_res: Annotated[Path, Product] = BLD
    / "bhatta"
    / "sims"
    / "combined_res.pkl",
) -> None:
    """Combine the simulation results."""
    df_to_concat = []

    # For each file in paths_to_sim_res, load the data frame
    for path in paths_to_sim_res:
        with Path.open(path, "rb") as file:
            res = pickle.load(file)
            _df = res["res"]
            _params = res["params"]

            # Add the parameters to the data frame
            for key, value in _params.items():
                if key == "c_n":
                    _df["c_n_name"] = str(value.keywords)
                    _df["c_n_val"] = value(_params["num_obs"])
                    _df["c_n_alpha"] = value.keywords["alpha"]
                    continue
                _df[key] = value

            df_to_concat.append(_df)

    out = pd.concat(df_to_concat)

    out["true"] = np.where(out["c_1"] < 0, out["c_1"], 0)

    out["covers_hi"] = out["ci_hi"] >= out["true"]
    out["covers_lo"] = out["ci_lo"] <= out["true"]

    out["covers"] = out["covers_hi"] & out["covers_lo"]

    out["ci_midpoint"] = (out["ci_hi"] + out["ci_lo"]) / 2

    out.to_pickle(path_to_combined_res)


for id_, kwargs in ID_TO_KWARGS_PLOTS.items():

    @pytask.mark.bhatta()
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_bhatta_sims(
        num_obs: int,
        stat_to_plot: str,
        path_to_plot: Annotated[Path, Product],
        path_to_combined_res: Path = BLD / "bhatta" / "sims" / "combined_res.pkl",
    ) -> None:
        """Description of the task."""
        data = pd.read_pickle(path_to_combined_res)

        data = data.query(f"num_obs == {num_obs}")

        fig = plot_bhatta_sims(data, stat_to_plot=stat_to_plot)

        fig.write_image(path_to_plot)


def plot_bhatta_sims(
    data: pd.DataFrame,
    stat_to_plot: str,
) -> go.Figure:
    """Plot the simulation results."""
    data_for_plot = data.groupby(["c_1", "c_n_name"]).mean()

    data_for_plot = data_for_plot.reset_index()

    setting = {}

    # Construct "c_n_alpha" by extracting the alpha from the c_n_name
    # c_n_name is of the form {'alpha': 0.05, 'sigma': 1.0}
    # So we need to extract the number following "alpha" from the string column

    for col in ["num_obs", "num_sims", "alpha"]:
        # Check unique, then assign to variable
        unique_values = data_for_plot[col].unique()

        if len(unique_values) != 1:
            msg = f"Multiple unique values for {col}"
            raise ValueError(msg)

        setting[col] = unique_values[0]

    c_n_alpha_to_color = {
        0.05: "blue",
        0.025: "red",
        0.0125: "green",
        0.1: "orange",
        0.2: "purple",
        0.4: "black",
        0.00625: "brown",
    }

    fig = go.Figure()

    for c_n_alpha in data_for_plot["c_n_alpha"].unique():
        data = data_for_plot.query(f"c_n_alpha == {c_n_alpha}")

        fig.add_trace(
            go.Scatter(
                x=data["c_1"],
                y=data[stat_to_plot],
                mode="lines",
                name=f"c_n alpha = {c_n_alpha}",
                line={"color": c_n_alpha_to_color[c_n_alpha]},
            ),
        )

    fig.update_layout(
        title=(
            f"Coverage of Confidence Intervals <br>"
            f"<sub>(N = {setting['num_obs']:.0f},"
            f" Simulations = {setting['num_sims']:.0f},"
            f" Nominal Coverage = {1 - setting['alpha']:.2f})</sub>"
        ),
        xaxis_title="True Value",
        yaxis_title=f"{stat_to_plot.capitalize()}",
    )

    c_1_min = np.min(data_for_plot["c_1"])
    c_1_max = np.max(data_for_plot["c_1"])

    # Add a line at 0.95
    if stat_to_plot in ["covers", "covers_hi", "covers_lo"]:
        fig.add_shape(
            type="line",
            x0=c_1_min,
            y0=1 - setting["alpha"],
            x1=c_1_max,
            y1=1 - setting["alpha"],
            line={"color": "black", "width": 1},
        )

    if stat_to_plot in ["v_hat", "ci_midpoint", "ci_lo", "ci_hi"]:
        fig.add_trace(
            go.Scatter(
                x=data_for_plot["c_1"],
                y=data_for_plot["true"],
                mode="lines",
                name="True v",
                line={"dash": "dash", "color": "black"},
            ),
        )

    if stat_to_plot in ["ci_midpoint"]:
        fig.add_trace(
            go.Scatter(
                x=data_for_plot["c_1"],
                y=data_for_plot["v_hat"],
                mode="lines",
                name="Estimated v",
                line={"dash": "dash", "color": "black"},
            ),
        )

    return fig
