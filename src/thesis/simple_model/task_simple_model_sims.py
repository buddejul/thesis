"""Task for running bootstrap simulations."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pytask import Product, task

from thesis.classes import Instrument, LocalATEs
from thesis.config import BLD, RNG
from thesis.simple_model.funcs import simulation_bootstrap
from thesis.utilities import get_func_as_string


class _Arguments(NamedTuple):
    u_hi: float
    path_to_data: Path
    pscore_low: float
    n_obs: int
    local_ates: LocalATEs
    constraint_mtr: str
    bootstrap_method: str
    bootstrap_params: dict[str, Callable]
    pscore_hi: float = 0.6
    alpha: float = 0.05
    n_boot: int = 2
    n_sims: int = 2
    rng: np.random.Generator = RNG


U_HI = [0.2]
N_OBS = [250, 1_000, 10_000]
PSCORES_LOW = [0.4]
CONSTRAINTS_MTR = ["increasing"]
BOOTSTRAP_METHODS = ["standard", "numerical_delta"]
LATES_COMPLIER = np.concat((np.linspace(-0.1, 0.1, num=10), np.zeros(1)))
EPS_FUNS_NUMERICAL_DELTA = [
    lambda n: n ** (-1 / 2),
    lambda n: n ** (-1 / 3),
    lambda n: n ** (-1 / 6),
]


# TODO(@buddejul): An alternative way to do this would be to simply give this a
# non-meaningful name (e.g. simple_models_sims_`i`) and store all params in the dataset
# or a separate dict (probably more efficient). We can then retrieve the params from the
# dataset or dict after saving. We can then query the settings dict before merging.
ID_TO_KWARGS = {
    (
        f"bootstrap_sims_{u_hi}_n_obs_{n_obs}_pscore_low_{pscore_low}"
        f"_late_complier_{late_complier}_constraint_mtr_{constraint_mtr}"
        f"bootstrap_method_{bootstrap_method}"
        f"_eps_fun_{get_func_as_string(eps_fun)}"
    ): _Arguments(
        u_hi=u_hi,
        n_obs=n_obs,
        pscore_low=pscore_low,
        local_ates=LocalATEs(
            never_taker=0,
            complier=late_complier,
            # The following ensures that the model is always correctly specified under
            # increasing MTR functions: Whenever late_complier < 0, the largest possible
            # always-taker ATE is 1 + later_complier < 1.
            always_taker=np.min((1, 1 + late_complier)),
        ),
        constraint_mtr=constraint_mtr,
        bootstrap_method=bootstrap_method,
        bootstrap_params={"eps_fun": eps_fun},
        path_to_data=Path(
            BLD
            / "boot"
            / "results"
            / (
                f"data_{u_hi}_n_obs_{n_obs}_pscore_low_{pscore_low}"
                f"_late_complier_{late_complier}_constraint_mtr_{constraint_mtr}"
                f"_bootstrap_method_{bootstrap_method}"
                f"_eps_fun_{get_func_as_string(eps_fun)}.pkl"
            ),
        ),
    )
    for u_hi in U_HI
    for n_obs in N_OBS
    for pscore_low in PSCORES_LOW
    for late_complier in LATES_COMPLIER
    for constraint_mtr in CONSTRAINTS_MTR
    for bootstrap_method in BOOTSTRAP_METHODS
    for eps_fun in EPS_FUNS_NUMERICAL_DELTA
    if not (
        bootstrap_method == "standard" and eps_fun is not EPS_FUNS_NUMERICAL_DELTA[0]
    )
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
        local_ates: LocalATEs,
        constraint_mtr: str,
        bootstrap_method: str,
        bootstrap_params: dict[str, Callable],
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
            local_ates=local_ates,
            alpha=alpha,
            instrument=instrument,
            constraint_mtr=constraint_mtr,
            rng=rng,
            bootstrap_method=bootstrap_method,
            bootstrap_params=bootstrap_params,
        )

        res.to_pickle(path_to_data)
