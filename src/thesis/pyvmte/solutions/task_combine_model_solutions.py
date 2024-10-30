"""Combine all model solutions into a single DataFrame."""

import pickle
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD
from thesis.pyvmte.solutions.task_solve_simple_model import ID_TO_KWARGS

paths_to_res = [ID_TO_KWARGS[key].path_to_data for key in ID_TO_KWARGS]

paths_to_full_solutions = [ID_TO_KWARGS[key].path_to_dicts for key in ID_TO_KWARGS]


# --------------------------------------------------------------------------------------
# Upper and Lower Bounds in DataFrame
# --------------------------------------------------------------------------------------
@pytask.mark.solve
def task_solutions_binary_iv_upper_lower_bounds(
    path_to_combined: Annotated[Path, Product] = BLD
    / "data"
    / "solutions"
    / "solutions_simple_model_combined.pkl",
    paths_to_res: list[Path] = paths_to_res,
) -> pd.DataFrame:
    """Combine model solutions into a single DataFrame."""
    dfs_single = [pd.read_pickle(path) for path in paths_to_res]

    df_combined = pd.concat(dfs_single, ignore_index=True)

    df_combined["b_late"] = df_combined["y1_c"] - df_combined["y0_c"]

    df_combined.to_pickle(path_to_combined)


# --------------------------------------------------------------------------------------
# Upper and Lower Bounds with solution coefficients in DataFrame
# --------------------------------------------------------------------------------------


@pytask.mark.solve
def task_solutions_binary_iv_full_results(
    path_to_combined: Annotated[Path, Product] = BLD
    / "data"
    / "solutions"
    / "full_solutions_simple_model_combined.pkl",
    paths_to_full_solutions: list[Path] = paths_to_full_solutions,
) -> pd.DataFrame:
    """Combine model solutions with coefficients into a single DataFrame."""
    df_to_append: list[pd.DataFrame] = []

    for path in paths_to_full_solutions:
        params_names = [
            "bfunc_type",
            "idestimands",
            "constraint_type",
            "constraint_val",
            "k_bernstein",
            "num_gridpoints",
        ]

        with Path.open(path, "rb") as file:
            res = pickle.load(file)

        # Separate the dictionary into params and results
        params = {key: res[key] for key in params_names}
        results = {key: res[key] for key in res if key not in params}

        # ------------------------------------------------------------------------------
        # Upper and lower bounds
        # ------------------------------------------------------------------------------
        upper_bounds = [results[key].upper_bound for key in results]
        lower_bounds = [results[key].lower_bound for key in results]

        complier_late = list(results.keys())

        # ------------------------------------------------------------------------------
        lower_x = [results[key].lower_optres.x for key in results]
        upper_x = [results[key].upper_optres.x for key in results]

        # ------------------------------------------------------------------------------
        # Put together DataFrame
        # ------------------------------------------------------------------------------

        out = pd.DataFrame(
            {
                "complier_late": complier_late,
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds,
            },
        )
        max_len_lower = max([len(x) for x in lower_x if x is not None])
        max_len_upper = max([len(x) for x in upper_x if x is not None])

        # Concatenate lower_x to array: lower_x[i] if not None else np.nan of length
        # max_len_lower
        x_lower = np.zeros((len(lower_x), max_len_lower))

        for i, x in enumerate(lower_x):
            if x is not None:
                x_lower[i, :] = x
            else:
                x_lower[i, :] = np.nan

        x_upper = np.zeros((len(upper_x), max_len_upper))

        for i, x in enumerate(upper_x):
            if x is not None:
                x_upper[i, :] = x
            else:
                x_upper[i, :] = np.nan

        for i in range(max_len_lower):
            out[f"x_lower_{i}"] = x_lower[:, i]

        for i in range(max_len_upper):
            out[f"x_upper_{i}"] = x_upper[:, i]

        out = out.set_index("complier_late")

        # Merge with params
        out = out.merge(
            pd.DataFrame(params, index=out.index),
            left_index=True,
            right_index=True,
        )

        df_to_append.append(out)

    df_combined = pd.concat(df_to_append)

    df_combined.to_pickle(path_to_combined)
