"""Plot distributions in Bhatta 2009 problem."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task

from thesis.bhattacharya.bhatta_funcs import plot_scaled_distr
from thesis.config import BLD, RNG


class _Arguments(NamedTuple):
    path_to_plot: Path
    num_obs: int
    num_reps: int
    num_grid: int
    c_1: float
    sigma: float
    rng: np.random.Generator = RNG


num_obs_to_plot = [100, 1_000, 10_000]
num_reps = 25_000
num_grid = 1_000
sigma = 1
c_1_to_plot = [-0.01, 0, 0.01]


ID_TO_KWARGS = {
    f"{num_obs}_{c_1}": _Arguments(
        path_to_plot=BLD / "bhatta" / "plots" / f"distr_{num_obs}_c_1_{c_1}.png",
        num_obs=num_obs,
        num_reps=num_reps,
        num_grid=num_grid,
        c_1=c_1,
        sigma=sigma,
    )
    for num_obs in num_obs_to_plot
    for c_1 in c_1_to_plot
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.bhatta()
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_simulate_distributions(
        num_obs: int,
        num_reps: int,
        num_grid: int,
        c_1: float,
        sigma: float,
        rng: np.random.Generator,
        path_to_plot: Annotated[Path, Product],
    ) -> None:
        """Description of the task."""
        fig = plot_scaled_distr(
            num_obs=num_obs,
            num_reps=num_reps,
            num_grid=num_grid,
            c_1=c_1,
            sigma=sigma,
            rng=rng,
        )

        fig.write_image(path_to_plot)
