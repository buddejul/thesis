"""Task for combining simulation results into dataset."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD
from thesis.simple_model.task_simple_model_sims import (
    EPS_FUNS_NUMERICAL_DELTA,
    ID_TO_KWARGS,
    _Arguments,
)
from thesis.utilities import get_func_as_string

EPS_FUNS_STRINGS = [get_func_as_string(eps_fun) for eps_fun in EPS_FUNS_NUMERICAL_DELTA]
KAPPA_FUNS_STRINGS = [
    get_func_as_string(kappa_fun) for kappa_fun in EPS_FUNS_NUMERICAL_DELTA
]


def task_combine_sim_results(
    id_to_kwargs: dict[str, _Arguments] = ID_TO_KWARGS,
    path_to_data: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "sim_results_combined.pkl",
    ),
) -> None:
    """Combine all simulation results into single dataframe."""
    coverage = []

    for kwargs in id_to_kwargs.values():
        res = pd.read_pickle(kwargs.path_to_data)  # noqa: S301
        res["ci_covers_true_param"] = (res["lo"] <= res["true"]) & (
            res["hi"] >= res["true"]
        )
        res["ci_length"] = res["hi"] - res["lo"]
        coverage.append(
            (
                kwargs.u_hi,
                kwargs.n_obs,
                kwargs.local_ates.complier,
                kwargs.bootstrap_method,
                get_func_as_string(kwargs.bootstrap_params["eps_fun"]),
                get_func_as_string(kwargs.bootstrap_params["kappa_fun"]),
                kwargs.constraint_mtr,
                res["true"].mean(),
                res["ci_covers_true_param"].mean(),
                res["ci_length"].mean(),
                res["lo"].mean(),
                res["hi"].mean(),
            ),
        )

    cols = [
        "u_hi",
        "n_obs",
        "late_complier",
        "bootstrap_method",
        "eps_fun",
        "kappa_fun",
        "constraint_mtr",
        "true",
        "coverage",
        "length",
        "ci_lo",
        "ci_hi",
    ]

    data = pd.DataFrame(coverage, columns=cols)

    data.to_pickle(path_to_data)
