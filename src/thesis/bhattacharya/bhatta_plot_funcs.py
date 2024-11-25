"""Functions for plotting Bhatta 2009 simulation results."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]


def plot_bhatta_sims(
    data: pd.DataFrame,
    stat_to_plot: str,
) -> go.Figure:
    """Plot the simulation results."""
    data_for_plot = data.groupby(["c_1", "c_n_name"]).mean()

    data_for_plot = data_for_plot.reset_index()

    setting = {}

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


def plot_bhatta_sims_histogram(
    data: pd.DataFrame,
    stat_to_plot: str,
    c_1: float,
    c_n_alpha: str,
    num_obs: int,
) -> go.Figure:
    """Plot the simulation results: histograms."""
    data_for_plot = data.query(
        f"c_1 == {c_1} & c_n_alpha == {c_n_alpha} & num_obs == {num_obs}",
    )

    if data_for_plot.empty:
        msg = f"No data for c_1 = {c_1}, c_n_alpha = {c_n_alpha}, num_obs = {num_obs}"
        raise ValueError(msg)

    data_for_plot = data_for_plot.reset_index()

    setting = {}

    for col in ["num_obs", "num_sims", "alpha"]:
        # Check unique, then assign to variable
        unique_values = data_for_plot[col].unique()

        if len(unique_values) > 1:
            msg = f"Multiple unique values for {col}"
            raise ValueError(msg)

        setting[col] = unique_values[0]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data_for_plot[stat_to_plot],
            name=stat_to_plot,
            histnorm="probability",
            marker_color="blue",
        ),
    )

    fig.update_layout(
        title=(
            f"Distribution of {stat_to_plot} <br>"
            f"<sub>(N = {setting['num_obs']:.0f},"
            f" Simulations = {setting['num_sims']:.0f},"
            f" Nominal Coverage = {1 - setting['alpha']:.2f})</sub>"
            f" c_1 = {c_1}, c_n = {c_n_alpha}"
        ),
        xaxis_title=stat_to_plot,
        yaxis_title="Density",
    )

    return fig
