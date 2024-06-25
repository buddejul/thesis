"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pytask import Product, task

from thesis.bootstrap_fail.funcs import simulation_bootstrap
from thesis.config import BLD, RNG


class _Arguments(NamedTuple):
    u_hi: float
    path_to_data: Path
    alpha: float = 0.05
    n_obs: int = 100
    n_boot: int = 2_000
    n_sims: int = 2_000
    rng: np.random.Generator = RNG


U_HI = np.linspace(0.60, 0.65, num=10)
N_OBS = [25, 100, 250]

ID_TO_KWARGS = {
    f"bootstrap_sims_{u_hi}_n_obs_{n_obs}": _Arguments(
        u_hi=u_hi,
        n_obs=n_obs,
        path_to_data=Path(BLD / "boot" / f"data_{u_hi}_n_obs_{n_obs}.pkl"),
    )
    for u_hi in U_HI
    for n_obs in N_OBS
}


for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_bootstrap_sims(
        n_sims: int,
        n_obs: int,
        n_boot: int,
        u_hi: float,
        alpha: float,
        rng: np.random.Generator,
        path_to_data: Annotated[Path, Product],
    ) -> None:
        """Task for running bootstrap simulations."""
        res = simulation_bootstrap(
            n_sims=n_sims,
            n_obs=n_obs,
            n_boot=n_boot,
            u_hi=u_hi,
            alpha=alpha,
            rng=rng,
            param_pos="boundary",
        )

        res.to_pickle(path_to_data)
