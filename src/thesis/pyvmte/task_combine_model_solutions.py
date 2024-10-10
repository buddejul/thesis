"""Combine all model solutions into a single DataFrame."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product

from thesis.config import BLD
from thesis.pyvmte.task_solve_simple_model import ID_TO_KWARGS

paths_to_res = [ID_TO_KWARGS[key].path_to_data for key in ID_TO_KWARGS]


@pytask.mark.pyvmte_sols
def task_combine_model_solutions_simple_model(
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
