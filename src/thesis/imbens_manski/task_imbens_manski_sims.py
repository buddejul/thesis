"""Run simulations for Imbens and Manski (2004) ECMA."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pytask import Product, task

from thesis.config import BLD, RNG
from thesis.imbens_manski.funcs import simulation


class _Arguments(NamedTuple):
    p: float
    n_obs: int
    n_sims: int
    path_to_results: Path
    alpha: float = 0.95
    beta_params: tuple[float, float] = (0.5, 0.5)


P_VALUES = np.linspace(0.9, 0.99, 100)
N_OBS_VALUES = [25, 50, 100]
N_SIMS = 1_000

ID_TO_KWARGS = {
    f"p_{p}_n_obs_{n_obs}": _Arguments(
        p=p,
        n_obs=n_obs,
        n_sims=N_SIMS,
        path_to_results=BLD / f"p_{p}_n_obs_{n_obs}.pkl",
    )
    for p in P_VALUES
    for n_obs in N_OBS_VALUES
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_manski_imbens_sims(
        p: float,
        n_obs: int,
        n_sims: int,
        alpha: float,
        beta_params: tuple[float, float],
        path_to_results: Annotated[Path, Product],
    ) -> None:
        """Run simulations for Imbens and Manski (2004) ECMA."""
        res = simulation(
            n_sims,
            n_obs,
            p,
            alpha=alpha,
            beta_params=beta_params,
            rng=RNG,
        )
        res.to_pickle(path_to_results)
