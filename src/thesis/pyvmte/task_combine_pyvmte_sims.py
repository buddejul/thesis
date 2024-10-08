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
@pytask.mark.skip()
def task_combine_pyvmte_sims(
    res_files: list[Path] = RES_FILES,
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "pyvmte_simulations" / "combined.pkl"
    ),
):
    """Combine pyvmte simulation results."""
    df_combined = pd.concat([pd.read_pickle(f) for f in res_files])

    df_combined.to_pickle(path_to_combined)
