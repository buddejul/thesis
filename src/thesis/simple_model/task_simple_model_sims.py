"""Task for running bootstrap simulations."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pytask import Product, task

from thesis.classes import Instrument, LocalATEs
from thesis.config import BLD, RNG
from thesis.simple_model.funcs import simulation_bootstrap


class _Arguments(NamedTuple):
    u_hi: float
    pscore_low: float
    n_obs: int
    local_ates: LocalATEs
    constraint_mtr: str
    bootstrap_method: str
    bootstrap_params: dict[str, Callable]
    path_to_data: Path | None = None
    pscore_hi: float = 0.6
    alpha: float = 0.05
    n_boot: int = 250
    n_sims: int = 250
    rng: np.random.Generator = RNG


U_HI = [0.2]
N_OBS = [1_000, 10_000]
PSCORES_LOW = [0.4]
CONSTRAINTS_MTR = ["increasing"]
BOOTSTRAP_METHODS = ["standard", "numerical_delta", "analytical_delta"]
LATES_COMPLIER = np.concat((np.linspace(-0.4, 0.4, num=12), np.zeros(1)))
EPS_FUNS_NUMERICAL_DELTA = [
    lambda n: n ** (-1 / 2),
]
KAPPA_FUNS_ANALYTICAL_DELTA = [
    lambda n: n ** (1 / 2),
    lambda n: np.log(n) ** (1 / 2),
]


# TODO(@buddejul): An alternative way to do this would be to simply give this a
# non-meaningful name (e.g. simple_models_sims_`i`) and store all params in the dataset
# or a separate dict (probably more efficient). We can then retrieve the params from the
# dataset or dict after saving. We can then query the settings dict before merging.
KWARGS = [
    _Arguments(
        u_hi=u_hi,
        n_obs=n_obs,
        pscore_low=pscore_low,
        local_ates=LocalATEs(
            always_taker=0,
            complier=late_complier,
            # The following ensures that the model is always correctly specified under
            # increasing MTR functions: Whenever late_complier < 0, the largest possible
            # always-taker ATE is 1 + later_complier < 1.
            never_taker=np.min((1, 1 + late_complier)),
        ),
        constraint_mtr=constraint_mtr,
        bootstrap_method=bootstrap_method,
        bootstrap_params={"eps_fun": eps_fun, "kappa_fun": kappa_fun},
    )
    for u_hi in U_HI
    for n_obs in N_OBS
    for pscore_low in PSCORES_LOW
    for late_complier in LATES_COMPLIER
    for constraint_mtr in CONSTRAINTS_MTR
    for bootstrap_method in BOOTSTRAP_METHODS
    for eps_fun in EPS_FUNS_NUMERICAL_DELTA
    for kappa_fun in KAPPA_FUNS_ANALYTICAL_DELTA
    # For standard bootstrap, we only need to run the simulation once
    if not (
        bootstrap_method == "standard"
        and (
            eps_fun is not EPS_FUNS_NUMERICAL_DELTA[0]
            or kappa_fun is not KAPPA_FUNS_ANALYTICAL_DELTA[0]
        )
    )
    # For the analytical delta method, we only need to run the simulation once for each
    # kappa_fun, but not for different eps_fun
    if not (
        bootstrap_method == "analytical_delta"
        and eps_fun is not EPS_FUNS_NUMERICAL_DELTA[0]
    )
    # For the numerical delta method, we only need to run the simulation once for each
    # eps_fun, but not for different kappa_fun
    if not (
        bootstrap_method == "numerical_delta"
        and kappa_fun is not KAPPA_FUNS_ANALYTICAL_DELTA[0]
    )
]

ID_TO_KWARGS = {str(id_): kwargs for id_, kwargs in enumerate(KWARGS)}

for id_, kwargs in ID_TO_KWARGS.items():
    ID_TO_KWARGS[id_] = kwargs._replace(
        path_to_data=BLD / "simple_model" / kwargs.bootstrap_method / f"{id_}.pkl",
    )

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
