"""Task for simulations using pyvmte."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task
from pyvmte.config import IV_SM  # type: ignore[import-untyped]

from thesis.config import BLD
from thesis.pyvmte.pyvmte_sims import simulation_pyvmte

# --------------------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------------------
num_sims = 20

u_hi_extra = 0.2

num_grid_points_complier_late = 20

confidence_intervals_to_sim = ["bootstrap", "subsampling"]

num_obs_to_sim = [1_000, 10_000]
bfuncs_to_sim = [
    # "constant",
    "bernstein",
]
id_estimands_to_sim = ["sharp"]

alpha = 0.05

confidence_interval_options = {
    "n_boot": 2_000,
    "n_subsamples": 2_000,
    "subsample_size": lambda n: 0.1 * n,
    "alpha": alpha,
}

k_bernstein = 11

shape_constraints = ("decreasing", "decreasing")
mte_monotone = "decreasing"
monotone_response = "positive"

constraints_to_sim = {
    "mte_monotone": mte_monotone,
    "monotone_response": monotone_response,
}

instrument = IV_SM


# --------------------------------------------------------------------------------------
# Construct inputs
# --------------------------------------------------------------------------------------
class _Arguments(NamedTuple):
    confidence_interval: str
    num_obs: int
    idestimands: str
    constraints: dict
    bfunc_type: str
    path_to_data: Annotated[Path, Product]
    complier_late: float
    num_sims: int = num_sims
    u_hi_extra: float = u_hi_extra
    confidence_interval_options: dict = confidence_interval_options


ID_TO_KWARGS = {
    (
        f"{bfunc}_{idestimands}_{constraint}_{confidence_interval}"
        f"_complier_late_{complier_late}_num_obs_{num_obs}"
    ): _Arguments(
        num_obs=num_obs,
        idestimands=idestimands,
        bfunc_type=bfunc,
        path_to_data=BLD
        / "data"
        / "pyvmte_simulations"
        / (
            "res_"
            f"{bfunc}_{idestimands}_{constraint}_{confidence_interval}"
            f"_complier_late_{complier_late}_num_obs_{num_obs}.pkl"
        ),
        constraints={k: constraints_to_sim[k] for k in [constraint]},
        confidence_interval=confidence_interval,
        complier_late=complier_late,
    )
    for num_obs in num_obs_to_sim
    for bfunc in bfuncs_to_sim
    for idestimands in id_estimands_to_sim
    for constraint in constraints_to_sim
    for confidence_interval in confidence_intervals_to_sim
    for complier_late in np.linspace(0, 1, num_grid_points_complier_late)
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.wip
    def task_pyvmte_simulations(
        num_sims: int,
        num_obs: int,
        path_to_data: Annotated[Path, Product],
        constraints: dict,
        bfunc_type: str,
        idestimands: str,
        complier_late: float,
        u_hi_extra: float,
        confidence_interval: str,
        confidence_interval_options: dict,
    ) -> None:
        """Perform simulations for the simple model using pyvmte."""
        # TODO(@buddejul): For now do this for a fixed set of parameters.

        tolerance_est = 5 / np.sqrt(num_obs)

        res = simulation_pyvmte(
            num_sims=num_sims,
            num_obs=num_obs,
            instrument=instrument,
            idestimands=idestimands,
            bfunc_type=bfunc_type,
            bfunc_options={"k_degree": k_bernstein},
            constraints=constraints,
            complier_late=complier_late,
            u_hi_extra=u_hi_extra,
            confidence_interval=confidence_interval,
            confidence_interval_options=confidence_interval_options,
            tolerance_est=tolerance_est,
        )

        res.to_pickle(path_to_data)
