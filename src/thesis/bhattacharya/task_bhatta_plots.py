"""Tasks for plotting Bhattachayra 2009 simulations."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from thesis.bhattacharya.bhatta_plot_funcs import (
    plot_bhatta_sims,
    plot_bhatta_sims_histogram,
)
from thesis.bhattacharya.task_bhatta_sims import ID_TO_KWARGS
from thesis.config import BLD

paths_to_sim_res = [kwargs.path_to_res for kwargs in ID_TO_KWARGS.values()]


class _ArgsPlots(NamedTuple):
    num_obs: int
    stat_to_plot: str
    path_to_plot: Path


class _ArgsPlotsHisto(NamedTuple):
    num_obs: int
    stat_to_plot: str
    c_1: float
    c_n_alpha: str
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


ID_TO_KWARGS_PLOTS_HISTO = {
    f"histo_{num_obs}_{stat_to_plot}_{c_1}_{c_n_alpha}": _ArgsPlotsHisto(
        num_obs=num_obs,
        stat_to_plot=stat_to_plot,
        c_1=c_1,
        c_n_alpha=c_n_alpha,
        path_to_plot=BLD
        / "bhatta"
        / "plots"
        / "histograms"
        / f"{stat_to_plot}"
        / f"histogram_{stat_to_plot}_{num_obs}_{c_1}_{c_n_alpha}.png",
    )
    for num_obs in [100, 1_000, 10_000]
    for stat_to_plot in ["z_lo", "z_hi"]
    for c_1 in [-0.01]
    for c_n_alpha in np.array([1, 8]) * 0.05
}


for id_histo, kwargs_histo in ID_TO_KWARGS_PLOTS_HISTO.items():

    @pytask.mark.bhatta()
    @task(id=id_histo, kwargs=kwargs_histo)  # type: ignore[arg-type]
    def task_plot_bhatta_sims_histogram(
        num_obs: int,
        stat_to_plot: str,
        c_1: float,
        c_n_alpha: str,
        path_to_plot: Annotated[Path, Product],
        path_to_combined_res: Path = BLD / "bhatta" / "sims" / "combined_res.pkl",
    ) -> None:
        """Description of the task."""
        data = pd.read_pickle(path_to_combined_res)

        fig = plot_bhatta_sims_histogram(data, stat_to_plot, c_1, c_n_alpha, num_obs)

        fig.write_image(path_to_plot)
