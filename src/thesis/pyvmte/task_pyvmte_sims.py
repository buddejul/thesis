"""Task for simulations using pyvmte."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task
from pyvmte.config import IV_SM  # type: ignore[import-untyped]

from thesis.classes import LocalATEs
from thesis.config import BLD
from thesis.pyvmte.pyvmte_sims import simulation_pyvmte

# --------------------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------------------
confidence_intervals = ["bootstrap", "rescaled_bootstrap", "subsampling"]

alpha = 0.05

confidence_interval_options = {
    "n_boot": 100,
    "n_subsamples": 100,
    "subsample_size": lambda n: 0.5 * n,
    "alpha": alpha,
}

k_bernstein = 11

shape_constraints = ("decreasing", "decreasing")
mte_monotone = "decreasing"
monotone_response = "positive"

constraints = {
    "shape_constraints": shape_constraints,
    "mte_monotone": mte_monotone,
    "monotone_response": monotone_response,
}

# TODO(@buddejul): Think about a way to do this better. The nt/at parameters are cur-
# rently hard-coded in the simulation function.
local_ates = LocalATEs(
    complier=0.5,
    always_taker=np.nan,
    never_taker=np.nan,
)

# --------------------------------------------------------------------------------------
# Construct inputs
# --------------------------------------------------------------------------------------


instrument = IV_SM


class _Arguments(NamedTuple):
    confidence_interval: str
    num_obs: int
    idestimands: str
    constraints: dict
    bfunc_type: str
    path_to_data: Annotated[Path, Product]
    num_sims: int = 2
    u_hi_extra: float = 0.2
    local_ates: LocalATEs = local_ates
    confidence_interval_options: dict = confidence_interval_options


ID_TO_KWARGS = {
    f"{bfunc}_{idestimands}_{constraint}_{confidence_interval}": _Arguments(
        num_obs=num_obs,
        idestimands=idestimands,
        bfunc_type=bfunc,
        path_to_data=BLD
        / "data"
        / "solutions"
        / (
            "solution_simple_model_"
            f"{bfunc}_{idestimands}_{constraint}_{confidence_interval}"
        ),
        constraints={k: constraints[k] for k in [constraint]},
        confidence_interval=confidence_interval,
    )
    for num_obs in [1000, 10_000]
    for bfunc in ["constant", "bernstein"]
    for idestimands in ["late", "sharp"]
    for constraint in constraints
    for confidence_interval in confidence_intervals
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.wip
    def task_solve_simple_model(
        num_sims: int,
        num_obs: int,
        path_to_data: Annotated[Path, Product],
        constraints: dict,
        bfunc_type: str,
        idestimands: str,
        local_ates: LocalATEs,
        u_hi_extra: float,
        confidence_interval: str,
        confidence_interval_options: dict,
    ) -> None:
        """Perform simulations for the simple model using pyvmte."""
        # TODO(@buddejul): For now do this for a fixed set of parameters.

        res = simulation_pyvmte(
            num_sims=num_sims,
            num_obs=num_obs,
            instrument=instrument,
            idestimands=idestimands,
            bfunc_type=bfunc_type,
            bfunc_options={"k_degree": k_bernstein},
            constraints=constraints,
            local_ates=local_ates,
            u_hi_extra=u_hi_extra,
            confidence_interval=confidence_interval,
            confidence_interval_options=confidence_interval_options,
        )

        res.to_pickle(path_to_data)
