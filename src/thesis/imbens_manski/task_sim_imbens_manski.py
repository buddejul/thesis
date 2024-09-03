"""Run simulations for Imbens and Manski (2004) ECMA."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task

from thesis.config import BLD, RNG
from thesis.imbens_manski.funcs import simulation


class _Arguments(NamedTuple):
    p: float
    n_obs: int
    n_sims: int
    path_to_results: Path
    ci_type: str
    alpha: float = 0.95
    beta_params: tuple[float, float] = (0.5, 0.5)


P_VALUES = np.linspace(0.9, 0.99, 25)
N_OBS_VALUES = [100, 250, 1000]
N_SIMS = 1_000
N_BOOT = 1_000
CI_TYPES = [
    "analytical",
    "bootstrap",
]

ID_TO_KWARGS = {
    f"p_{p}_n_obs_{n_obs}_{ci_type}": _Arguments(
        p=p,
        n_obs=n_obs,
        n_sims=N_SIMS,
        ci_type=ci_type,
        path_to_results=BLD
        / "imbens_manski"
        / "results"
        / f"p_{p}_n_obs_{n_obs}_{ci_type}.pkl",
    )
    for p in P_VALUES
    for n_obs in N_OBS_VALUES
    for ci_type in CI_TYPES
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.skip()
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_manski_imbens_sims(
        p: float,
        n_obs: int,
        n_sims: int,
        alpha: float,
        ci_type: str,
        beta_params: tuple[float, float],
        path_to_results: Annotated[Path, Product],
    ) -> None:
        """Run simulations for Imbens and Manski (2004) ECMA."""
        res = simulation(
            n_sims,
            n_obs,
            p,
            alpha=alpha,
            ci_type=ci_type,
            beta_params=beta_params,
            n_boot=N_BOOT,
            rng=RNG,
        )
        res.to_pickle(path_to_results)
