"""Combine results from pyvmte sims run on the HPC cluster.

Note that most simulations will be run on the cluster. Hence, this task should take a
folder with job outputs as inputs and then combine all files in these folders.

"""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD
from thesis.pyvmte.task_pyvmte_sims import ID_TO_KWARGS

HPC = False

if HPC is True:
    RES_DIRS = [
        BLD / "data" / "pyvmte_simulations",
    ]

    RES_FILES: list[Path] = []
    for d in RES_DIRS:
        RES_FILES.extend(d.glob("*.pkl"))

if HPC is False:
    RES_FILES = [v.path_to_data for v in ID_TO_KWARGS.values()]


# Skip for now: Combining only works with copied files else we have a cycles in the DAG.
@pytask.mark.wip
def task_combine_pyvmte_sims(
    res_files: list[Path] = RES_FILES,
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "pyvmte_simulations" / "combined.pkl"
    ),
):
    """Combine pyvmte simulation results."""
    df_to_combine = []

    for f in res_files:
        _df = pd.read_pickle(f)

        _df["covers_true_param"] = (_df["sim_ci_lower"] <= _df["true_param"]) & (
            _df["sim_ci_upper"] >= _df["true_param"]
        )

        _df["covers_idset_upper"] = _df["sim_ci_upper"] >= _df["true_upper_bound"]
        _df["covers_idset_lower"] = _df["sim_ci_lower"] <= _df["true_lower_bound"]
        _df["covers_idset"] = _df["covers_idset_upper"] & _df["covers_idset_lower"]

        # TODO: Would not work for multiple confidence_interval options.
        _to_collapse_by = [
            "bfunc_type",
            "idestimands",
            "num_obs",
            "u_hi_extra",
            "alpha",
            "confidence_interval",
            "k_bernstein",
            "shape_constraints",
            "mte_monotone",
            "monotone_response",
        ]

        _cols_to_collapse = [
            "sim_ci_lower",
            "sim_ci_upper",
            "sim_lower_bound",
            "sim_upper_bound",
            "true_lower_bound",
            "true_upper_bound",
            "true_param",
            "success_lower",
            "success_upper",
            "y1_c",
            "y0_c",
            "covers_true_param",
            "covers_idset",
            "covers_idset_upper",
            "covers_idset_lower",
            "num_sims",
            "n_boot",
            "n_subsamples",
            "subsample_size",
        ]

        _df = _df[_cols_to_collapse + _to_collapse_by]

        _groupby = _df.groupby(_to_collapse_by)

        _df_collapsed = _groupby[_cols_to_collapse].mean().reset_index()

        df_to_combine.append(_df_collapsed)

    df_combined = pd.concat(df_to_combine, ignore_index=True)

    df_combined["late_complier"] = df_combined["y1_c"] - df_combined["y0_c"]

    df_combined.to_pickle(path_to_combined)
