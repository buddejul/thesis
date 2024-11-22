"""Task for running Bhatta 2009 simulations."""
import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task

from thesis.bhattacharya.bhatta_funcs import c_n_normal, sim_confidence_interval
from thesis.config import BLD, RNG

params = {
    "alpha": 0.05,
    "num_sims": 1_000,
    "sigma": 1,
}


class _Arguments(NamedTuple):
    path_to_res: Path
    num_obs: int
    c_1: float
    c_n: Callable
    sigma: float = params["sigma"]
    alpha: float = params["alpha"]
    num_sims: float = params["num_sims"]
    rng: np.random.Generator = RNG


num_obs_for_sim = [100, 1_000]

num_gridpoints = 10
start = -0.2
end = 0.2
c_1_for_sim = np.sort(
    np.concatenate([np.linspace(start, end, num_gridpoints), np.zeros(1)]),
)

c_n_for_sim = [
    partial(
        c_n_normal,
        alpha=params["alpha"] / ratio,
        sigma=params["sigma"],
    )
    for ratio in [0.25, 0.5, 1, 2, 4]
]

ID_TO_KWARGS = {
    f"{num_obs}_{c_1}_{c_n(num_obs):.2f}": _Arguments(
        path_to_res=BLD / "bhatta" / "sims" / f"{num_obs}_{c_1}_{c_n(num_obs):.2f}.pkl",
        num_obs=num_obs,
        c_1=c_1,
        c_n=c_n,
    )
    for num_obs in num_obs_for_sim
    for c_1 in c_1_for_sim
    for c_n in c_n_for_sim
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.bhatta()
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_bhatta_sims(
        path_to_res: Annotated[Path, Product],
        num_obs: int,
        c_1: float,
        c_n: float,
        sigma: float,
        num_sims: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        """Description of the task."""
        res = sim_confidence_interval(
            num_obs=num_obs,
            c_1=c_1,
            c_n=c_n,
            sigma=sigma,
            num_sims=num_sims,
            alpha=alpha,
            rng=rng,
        )

        sim_params = {
            "num_obs": num_obs,
            "c_1": c_1,
            "c_n": c_n,
            "sigma": sigma,
            "alpha": alpha,
            "num_sims": num_sims,
        }

        out = {"params": sim_params, "res": res}

        with Path.open(path_to_res, "wb") as f:
            pickle.dump(out, f)
