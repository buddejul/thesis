"""Combine results from pyvmte sims run on the HPC cluster.

Note that most simulations will be run on the cluster. Hence, this task should take a
folder with job outputs as inputs and then combine all files in these folders.

"""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD, HPC_RES
from thesis.pyvmte._task_pyvmte_sims import ID_TO_KWARGS

HPC = True

JOBS = [
    17162646,
    17162699,
    17174486,
    17176830,
    17176989,
    17185050,
    17185642,
    17190151,
    17195484,
]

if HPC is True:
    RES_DIRS = [
        HPC_RES / f"{jobid}" / "bld" / "data" / "pyvmte_simulations" for jobid in JOBS
    ]

    RES_FILES: list[Path] = []
    for d in RES_DIRS:
        RES_FILES.extend(d.glob("*.pkl"))

if HPC is False:
    RES_FILES = [v.path_to_data for v in ID_TO_KWARGS.values()]


@pytask.mark.wip
def task_combine_pyvmte_sims(  # noqa: C901
    res_files: list[Path] = RES_FILES,
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "pyvmte_simulations" / "combined.pkl"
    ),
):
    """Combine pyvmte simulation results."""
    df_to_combine = []

    for f in res_files:
        _df = pd.read_pickle(f)

        # Skip all files without _iteration_ in the name
        if "_iteration_" not in f.name:
            continue

        # Get iteration from file name f, after _iteration_
        iteration = int(f.name.split("_iteration_")[1].split(".")[0])

        # Extract jobid from path, comes after "/marvin/"
        jobid = int(f.parts[f.parts.index("marvin") + 1])

        _df["jobid"] = jobid
        _df["iteration"] = iteration

        _df["covers_true_param"] = (_df["sim_ci_lower"] <= _df["true_param"]) & (
            _df["sim_ci_upper"] >= _df["true_param"]
        )

        _df["covers_idset_upper"] = _df["sim_ci_upper"] >= _df["true_upper_bound"]
        _df["covers_idset_lower"] = _df["sim_ci_lower"] <= _df["true_lower_bound"]
        _df["covers_idset"] = _df["covers_idset_upper"] & _df["covers_idset_lower"]

        # Whenever sim_ci_upper or sim_ci_lower is missing, set covers to False
        for col in [
            "covers_true_param",
            "covers_idset_upper",
            "covers_idset_lower",
            "covers_idset",
        ]:
            _df.loc[
                _df["sim_ci_upper"].isna() | _df["sim_ci_lower"].isna(),
                col,
            ] = False

        # breakpoint if "true_param" is about 0.5
        if np.isclose(_df["true_param"].mean(), 0.5, atol=0.1):
            pass

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
            "iteration",
            "jobid",
        ]

        _df = _df[_cols_to_collapse + _to_collapse_by]

        _groupby = _df.groupby(_to_collapse_by)

        _df_collapsed = _groupby[_cols_to_collapse].mean().reset_index()

        df_to_combine.append(_df_collapsed)

    df_combined = pd.concat(df_to_combine, ignore_index=True)

    df_combined["late_complier"] = df_combined["y1_c"] - df_combined["y0_c"]

    # Check if data is unique by _cols_to_collapse
    _cols_unique = [
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
        "late_complier",
    ]
    if df_combined.groupby([*_cols_unique, "iteration", "jobid"]).size().max() != 1:
        msg = "Not unique when 'iteration' and 'jobid' included."
        raise ValueError(msg)

    if df_combined.groupby(_cols_unique).size().max() != 1:
        _cols_to_average = [
            "sim_ci_lower",
            "sim_ci_upper",
            "sim_lower_bound",
            "sim_upper_bound",
            "covers_true_param",
            "covers_idset",
            "covers_idset_upper",
            "covers_idset_lower",
        ]
        # For each column in _cols_to_average compute the mean over iteraitons weighted
        # by num_sims. Compute total number of sims within each group
        # TODO(@buddejul): Something here is off.

        df_combined["total_sims"] = df_combined.groupby(_cols_unique)[
            "num_sims"
        ].transform("sum")

        # TODO: Might want to check at this point that coverage rates vary by iterations

        for c in _cols_to_average:
            df_combined[c] = (df_combined[c] * df_combined["num_sims"]) / df_combined[
                "total_sims"
            ]

            df_combined[c] = df_combined.groupby(_cols_unique)[c].transform("sum")
            # TODO(@buddejul): In some cases the simulated upper bounds might be
            # slightly greater than 1.
            _eps = 6e-2
            if not (df_combined[c] <= 1 + _eps).all():
                msg = f"Column {c} is greater than 1. Max: {df_combined[c].max()}"
                raise ValueError(msg)

            if "covers" in c:
                if not (df_combined[c] >= 0 - _eps).all():
                    msg = f"Column {c} is smaller than 0. Min: {df_combined[c].min()}"
                    raise ValueError(msg)

            elif not (df_combined[c] >= -1 - _eps).all():
                msg = f"Column {c} is smaller than -1. Min: {df_combined[c].min()}"
                raise ValueError(msg)

        df_combined["num_sims"] = df_combined["total_sims"]

        # Keep only unique rows
        df_combined = df_combined.drop_duplicates(subset=_cols_unique)

    # Assert unique now
    if df_combined.groupby(_cols_unique).size().max() != 1:
        msg = "Not unique after averaging over iterations."
        raise ValueError(msg)

    df_combined.to_pickle(path_to_combined)
