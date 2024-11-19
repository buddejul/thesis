"""Combine results from pyvmte sims run on the HPC cluster.

Note that most simulations will be run on the cluster. Hence, this task should take a
folder with job outputs as inputs and then combine all files in these folders.

"""

import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD, HPC_RES

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
    17699039,
    17700618,
    17777546,
    17778314,
    17782509,
    17782513,
    17783454,
    17784364,
]

JOBS_SQRT_TOLERANCE = [
    17699039,
    17700618,
    17777546,
    17778314,
]

JOBS_SQUARE_TOLERANCE = [
    17784364,
]


JOBS_IM_CRIT = [
    17783454,
]


# Create dictionary with jobid as key and tolerance as value
# tolerance is "1/sqrt(n)" if jobid in JOBS_SQRT_TOLERANCE
# tolerance is "1/n**2" if jobid in JOBS_SQUARE_TOLERANCE
# tolerance is "1/n" otherwise
def _get_tolerance(jobid):
    if jobid in JOBS_SQRT_TOLERANCE:
        return "1/sqrt(n)"
    if jobid in JOBS_SQUARE_TOLERANCE:
        return "1/n**2"

    return "1/n"


RES_FILES_TAR = [HPC_RES / f"{jobid}.tar.gz" for jobid in JOBS]
constant_cols = [
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
    "subsample_size",
    "num_sims",
    "n_boot",
    "n_subsamples",
    "alpha_im_crit",
]

cols_to_average = [
    "sim_ci_lower",
    "sim_ci_upper",
    "sim_lower_bound",
    "sim_upper_bound",
    "true_lower_bound",
    "true_upper_bound",
    "true_param",
    "alpha_for_ci",
]

cols_coverage = [
    "covers_true_param",
    "covers_idset_upper",
    "covers_idset_lower",
    "covers_idset",
]


@pytask.mark.wip
def task_combine_pyvmte_sims(
    jobs: list[int] = JOBS,
    path_to_combined_by_job: Annotated[Path, Product] = (
        BLD / "data" / "pyvmte_simulations" / "combined_by_job.pkl"
    ),
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "pyvmte_simulations" / "combined.pkl"
    ),
):
    """Combine pyvmte simulation results."""
    data_by_job = [extract_and_aggregate_result(job) for job in jobs]

    data_combined = pd.concat(data_by_job, ignore_index=True)

    data_combined.to_pickle(path_to_combined_by_job)

    # Take means by constant_cols, late_complier, num_obs, lp_tolerance
    cols_groupby = [*constant_cols, "late_complier", "lp_tolerance"]
    cols_groupby.remove("num_sims")

    def weighted_mean(x):
        return np.average(x, weights=data_combined.loc[x.index, "num_sims"])

    data_grouped = (
        data_combined.groupby(cols_groupby)
        .agg({col: weighted_mean for col in cols_to_average + cols_coverage})
        .reset_index()
    )

    data_num_sims = data_combined.groupby(cols_groupby)["num_sims"].sum().reset_index()

    data_grouped = data_grouped.merge(data_num_sims, on=cols_groupby)

    # TODO(@buddejul): Weight the means by num_sims.

    data_grouped["subsample_share"] = (
        data_grouped["subsample_size"] / data_grouped["num_obs"]
    )

    data_grouped.to_pickle(path_to_combined)


def extract_and_aggregate_result(jobid: int):
    """Extract and aggregate results from a single job."""
    tmp_dir = BLD / "marvin" / "_tmp"

    # Remove tmp_dir if it exists to not have old files in there by accident
    shutil.rmtree(tmp_dir, ignore_errors=True)

    file = HPC_RES / f"{jobid}.tar.gz"

    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(path=tmp_dir, filter="data")

    # Collect al lfile names in the temporary directory and all its subdirectories
    res_files = list(tmp_dir.rglob("*.pkl"))

    df_to_combine: list[pd.DataFrame] = []

    for f in res_files:
        # Skip all files without _simple_model_ in the name
        if "simple_model" in f.parts:
            warning = f"Found _simple_model in job {jobid}. Skipping."
            warnings.warn(warning, stacklevel=2)

            return pd.DataFrame()

        _df = pd.read_pickle(f)

        _df["jobid"] = jobid
        try:
            iteration = int(f.name.split("_iteration_")[1].split(".")[0])
        except IndexError:
            iteration = 0
        assert isinstance(iteration, int)

        _df["iteration"] = iteration

        # All early jobs where run with alpha_im_crit = False and alpha_for_ci = alpha
        if "alpha_for_ci" not in _df.columns:
            _df["alpha_for_ci"] = _df["alpha"]

        if "alpha_im_crit" not in _df.columns:
            _df["alpha_im_crit"] = False

        assert _df["true_param_pos"].unique() == "lower"

        cols_to_drop = ["success_upper", "success_lower", "true_param_pos"]
        _df = _df.drop(columns=cols_to_drop)

        # Separate the constant columns from the columns that will be averaged

        # ------------------------------------------------------------------------------
        # Compute Coverage
        # ------------------------------------------------------------------------------

        _df["covers_true_param"] = (_df["sim_ci_lower"] <= _df["true_param"]) & (
            _df["sim_ci_upper"] >= _df["true_param"]
        )

        _df["covers_idset_upper"] = _df["sim_ci_upper"] >= _df["true_upper_bound"]
        _df["covers_idset_lower"] = _df["sim_ci_lower"] <= _df["true_lower_bound"]
        _df["covers_idset"] = _df["covers_idset_upper"] & _df["covers_idset_lower"]

        # Whenever sim_ci_upper or sim_ci_lower is missing, set covers to False
        _df = _set_coverage_to_missing_if_sim_ci_missing(_df)
        _error_if_coverage_not_missing_if_sim_ci_missing(_df)

        # ------------------------------------------------------------------------------
        # Collapse over late_complier
        # ------------------------------------------------------------------------------
        _df["late_complier"] = _df["y1_c"] - _df["y0_c"]

        _df_grouped = (
            _df.groupby([*constant_cols, "late_complier"]).mean().reset_index()
        )

        df_to_combine.append(_df_grouped)

    df_combined = pd.concat(df_to_combine, ignore_index=True)

    cols_uniquely_sorted = [
        "late_complier",
        "iteration",
        "num_obs",
        "confidence_interval",
        "shape_constraints",
        "mte_monotone",
        "monotone_response",
    ]

    df_combined = df_combined.sort_values(cols_uniquely_sorted)

    df_combined["lp_tolerance"] = _get_tolerance(jobid)

    assert df_combined.set_index(cols_uniquely_sorted).index.is_unique

    return df_combined


def _error_if_coverage_not_missing_if_sim_ci_missing(df: pd.DataFrame):
    """Check if coverage is missing if sim_ci_upper or sim_ci_lower is missing."""
    cols_to_check = [
        "covers_true_param",
        "covers_idset_upper",
        "covers_idset_lower",
        "covers_idset",
    ]

    # Rows where sim_ci_upper or sim_ci_lower is missing
    idx_sim_ci_missing = df["sim_ci_upper"].isna() | df["sim_ci_lower"].isna()

    for col in cols_to_check:
        # Rows where coverage is missing
        idx_col_missing = df[col].isna()

        if (idx_sim_ci_missing & ~idx_col_missing).any():
            msg = (
                f"Coverage column {col} is non-missing for some rows where"
                "sim_ci_upper or sim_ci_lower is missing."
            )
            raise ValueError(
                (msg),
            )


def _set_coverage_to_missing_if_sim_ci_missing(df: pd.DataFrame):
    idx_sim_ci_missing = df["sim_ci_upper"].isna() | df["sim_ci_lower"].isna()

    cols_to_check = [
        "covers_true_param",
        "covers_idset_upper",
        "covers_idset_lower",
        "covers_idset",
    ]

    for col in cols_to_check:
        df.loc[idx_sim_ci_missing, col] = pd.NA

    return df
