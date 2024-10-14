"""Task for simulations using pyvmte."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product, task
from pyvmte.config import IV_SM  # type: ignore[import-untyped]

from thesis.config import BLD, NO_LONG_RUNNING_TASKS, RNG, SRC
from thesis.pyvmte.pyvmte_sims import simulation_pyvmte
from thesis.utilities import constraint_dict_to_string

# --------------------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------------------
# We perform num_sims * num_iterations simulations.
# The reason is to make tasks smaller to avoid long-running tasks on the cluster.
num_sims = 100
num_iterations = 6

u_hi_extra = 0.2

num_grid_late = 20

lo_grid = 0
hi_grid = 1

grid_late_complier = np.linspace(lo_grid, hi_grid, num_grid_late)

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

constraints_by_type: dict[str, dict] = {
    "none": {
        "shape_constraints": None,
        "mte_monotone": None,
        "monotone_response": None,
    },
    "mte_monotone": {
        "shape_constraints": None,
        "mte_monotone": mte_monotone,
        "monotone_response": None,
    },
    "monotone_response": {
        "shape_constraints": None,
        "mte_monotone": None,
        "monotone_response": monotone_response,
    },
}


instrument = IV_SM

# Define different grids to focus on interesting parameter intervals.
path_solution = SRC / "data" / "solutions" / "solutions_simple_model_combined.pkl"

sols = pd.read_pickle(path_solution)

sols = sols.query(
    'bfunc_type == "bernstein" and '
    'idestimands == "sharp" and '
    'constraint_type in ["none", "mte_monotone", "monotone_response"]',
).dropna(subset=["upper_bound", "lower_bound"])

# Check dataset is unique by: ["b_late", "constraint_type"]
sols = sols.set_index(["b_late", "constraint_type"])
assert sols.index.is_unique

b_late_min_max = (
    sols.reset_index().groupby("constraint_type")["b_late"].agg(["min", "max"])
)

_stop_mte_monotone = b_late_min_max.loc["mte_monotone", "max"]

grid_by_constraint = {
    "none": np.linspace(0.7, 1, num_grid_late),
    "mte_monotone": np.linspace(0.5, _stop_mte_monotone, num_grid_late),
    "monotone_response": np.linspace(0.65, 1, num_grid_late),
}

# Create a dictionary with constraints and late in grid for each type of constraint.
constraints_and_late_for_sim: list[tuple] = []

constraints_and_late_for_sim = [
    (constraints_by_type[constr], late)
    for constr in ["none", "mte_monotone", "monotone_response"]
    for late in grid_by_constraint[constr]
]


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
    true_param_pos: str = "lower"
    tolerance_est: float | None = None
    rng: np.random.Generator | None = None


ID_TO_KWARGS = {
    (
        f"{bfunc}_{idestimands}_{constraint_dict_to_string(constraint_dict)}_{confidence_interval}"
        f"_complier_late_{complier_late}_num_obs_{num_obs}_iteration_{iteration}"
    ): _Arguments(
        num_obs=num_obs,
        idestimands=idestimands,
        bfunc_type=bfunc,
        path_to_data=BLD
        / "data"
        / "pyvmte_simulations"
        / (
            "res_"
            f"{bfunc}_{idestimands}_"
            f"{constraint_dict_to_string(constraint_dict)}_"
            f"_{confidence_interval}"
            f"_complier_late_{complier_late}_num_obs_{num_obs}_iteration_{iteration}.pkl"
        ),
        constraints=constraint_dict,
        confidence_interval=confidence_interval,
        complier_late=complier_late,
    )
    for num_obs in num_obs_to_sim
    for bfunc in bfuncs_to_sim
    for idestimands in id_estimands_to_sim
    # Get the grid based on the key in constraint_dict with non-None value
    for constraint_dict, complier_late in constraints_and_late_for_sim
    for confidence_interval in confidence_intervals_to_sim
    for iteration in np.arange(num_iterations)
}

# --------------------------------------------------------------------------------------
# Set RNGs for each task
# --------------------------------------------------------------------------------------
# Avoid pitfalls with parallel processing by creating a separate RNG per task
# https://blog.scientific-python.org/numpy/numpy-rng/#parallel-processing
n_tasks = len(ID_TO_KWARGS)
child_rngs = RNG.spawn(n_tasks)

for (id_, kwargs), rng in zip(ID_TO_KWARGS.items(), child_rngs, strict=True):
    ID_TO_KWARGS[id_] = kwargs._replace(rng=rng)

# --------------------------------------------------------------------------------------
# Define tasks
# --------------------------------------------------------------------------------------
for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.hpc_pyvmte
    @pytask.mark.skipif(
        NO_LONG_RUNNING_TASKS,
        reason="Skip long-running tasks meant for cluster.",
    )
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
        true_param_pos: str,
        tolerance_est: float,
        rng: np.random.Generator,
    ) -> None:
        """Perform simulations for the simple model using pyvmte."""
        if tolerance_est is None:
            tolerance_est = 1 / num_obs

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
            true_param_pos=true_param_pos,
            rng=rng,
        )

        res["k_bernstein"] = k_bernstein

        res.to_pickle(path_to_data)
