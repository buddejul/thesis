"""Solve the simple model for a specification resulting in sharp bounds."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pytask import Product
from pyvmte.classes import Estimand, Instrument  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]

from thesis.config import BLD

# --------------------------------------------------------------------------------------
# Preliminary parameters
# --------------------------------------------------------------------------------------
num_gridpoints = 100

u_hi = 0.2

target = Estimand(
    "late",
    u_lo=0.4,
    u_hi=0.6 + u_hi,
)

identified = [
    Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
]

pscore_lo = 0.4
pscore_hi = 0.6

instrument = Instrument(
    support=np.array([0, 1]),
    pmf=np.array([0.5, 0.5]),
    pscores=np.array([pscore_lo, pscore_hi]),
)

bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": pscore_lo}
bfunc_2 = {"type": "constant", "u_lo": pscore_lo, "u_hi": pscore_hi}
bfunc_3 = {"type": "constant", "u_lo": pscore_hi, "u_hi": pscore_hi + u_hi}
bfunc_4 = {"type": "constant", "u_lo": pscore_hi + u_hi, "u_hi": 1.0}

bfuncs = [bfunc_1, bfunc_2, bfunc_3, bfunc_4]

u_partition = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi, 1])


def _at(u: float) -> bool:
    return u <= pscore_lo


def _c(u: float) -> bool:
    return pscore_lo <= u and pscore_hi > u


def _nt(u: float) -> bool:
    return u >= pscore_hi


# The following MTR functions are decreasing and imply a decreasing MTE function.
y1_at = 0.9
y0_at = 0

y1_nt = 0.5
y0_nt = 0.4


# Define function factories to avoid late binding
# See https://stackoverflow.com/a/3431699
def _make_m0(y0_c):
    def _m0(u):
        return y0_at * _at(u) + y0_c * _c(u) + y0_nt * _nt(u)

    return _m0


def _make_m1(y1_c):
    def _m1(u):
        return y1_at * _at(u) + y1_c * _c(u) + y1_nt * _nt(u)

    return _m1


def task_solve_simple_model_sharp(
    param_grid: np.ndarray | None,
    path_to_data: Annotated[Path, Product] = BLD
    / "data"
    / "solutions_simple_model_sharp.pkl",
    shape_constraint: tuple[str, str] = ("decreasing", "decreasing"),
) -> None:
    """Solve the simple model for a range of parameter values."""
    if param_grid is None:
        param_grid = np.linspace(0, 1, num_gridpoints)

    # Generate solution for a meshgrid of parameter values
    y1_c, y0_c = np.meshgrid(param_grid, param_grid)

    # Flatten the meshgrid
    y1_c_flat = y1_c.flatten()
    y0_c_flat = y0_c.flatten()

    results = []

    for y1_c_val, y0_c_val in zip(y1_c_flat, y0_c_flat, strict=True):
        _m1 = _make_m1(y1_c_val)
        _m0 = _make_m0(y0_c_val)

        # The identified set might be empty for some parameter value combinations.
        try:
            res = identification(
                target=target,
                identified_estimands=identified,
                basis_funcs=bfuncs,
                instrument=instrument,
                u_partition=u_partition,
                m0_dgp=_m0,
                m1_dgp=_m1,
                shape_constraints=shape_constraint,
            )
        except TypeError:
            res = {"upper_bound": np.nan, "lower_bound": np.nan}

        results.append(res)

    # Put into pandas DataFrame and save to disk

    df_res = pd.DataFrame(
        {
            "y1_c": y1_c_flat,
            "y0_c": y0_c_flat,
            "upper_bound": [r["upper_bound"] for r in results],
            "lower_bound": [r["lower_bound"] for r in results],
            "shape_constraint": [shape_constraint for _ in results],
            "y1_at": y1_at,
            "y0_at": y0_at,
            "y1_nt": y1_nt,
            "y0_nt": y0_nt,
        },
    )

    df_res.to_pickle(path_to_data)
