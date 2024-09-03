"""Task for running bootstrap simulations."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pytask import Product, task

from thesis.bootstrap_fail.funcs import simulation_bootstrap
from thesis.classes import Instrument, LocalATEs
from thesis.config import BLD, RNG


class _Arguments(NamedTuple):
    u_hi: float
    path_to_data: Path
    pscore_low: float
    n_obs: int
    pscore_hi: float = 0.6
    alpha: float = 0.05
    n_boot: int = 1_000
    n_sims: int = 1_000
    rng: np.random.Generator = RNG


U_HI = np.linspace(0, 0.05, num=10)
N_OBS = [100, 250]
PSCORES_LOW = np.linspace(0.55, 0.6, num=10)

LOCAL_ATES = LocalATEs(
    never_taker=0,
    complier=0.5,
    always_taker=1,
)

ID_TO_KWARGS = {
    f"bootstrap_sims_{u_hi}_n_obs_{n_obs}_pscore_low_{pscore_low}": _Arguments(
        u_hi=u_hi,
        n_obs=n_obs,
        pscore_low=pscore_low,
        path_to_data=Path(
            BLD
            / "boot"
            / "results"
            / f"data_{u_hi}_n_obs_{n_obs}_pscore_low_{pscore_low}.pkl",
        ),
    )
    for u_hi in U_HI
    for n_obs in N_OBS
    for pscore_low in PSCORES_LOW
}


for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_bootstrap_sims(
        n_sims: int,
        n_obs: int,
        n_boot: int,
        u_hi: float,
        alpha: float,
        pscore_low: float,
        pscore_hi: float,
        rng: np.random.Generator,
        path_to_data: Annotated[Path, Product],
    ) -> None:
        """Task for running bootstrap simulations."""
        instrument = Instrument(
            support=np.array([0, 1]),
            pmf=np.array([0.5, 0.5]),
            pscores=np.array([pscore_low, pscore_hi]),
        )

        res = simulation_bootstrap(
            n_sims=n_sims,
            n_obs=n_obs,
            n_boot=n_boot,
            u_hi=u_hi,
            local_ates=LOCAL_ATES,
            alpha=alpha,
            instrument=instrument,
            rng=rng,
        )

        res.to_pickle(path_to_data)
