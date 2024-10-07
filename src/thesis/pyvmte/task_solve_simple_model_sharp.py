"""Solve the simple model for a specification resulting in sharp bounds."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pytask import Product, task
from pyvmte.classes import Estimand  # type: ignore[import-untyped]
from pyvmte.config import IV_SM  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
    generate_constant_splines_basis_funcs,
)

from thesis.config import BLD

# --------------------------------------------------------------------------------------
# Preliminary parameters
# --------------------------------------------------------------------------------------
k_bernstein = 11

u_hi_extra = 0.2


identified_sharp = [
    Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
]

instrument = IV_SM
pscore_lo = instrument.pscores[0]
pscore_hi = instrument.pscores[1]

target = Estimand(
    "late",
    u_lo=pscore_lo,
    u_hi=pscore_hi,
    u_hi_extra=u_hi_extra,
)

identified_late = [Estimand(esttype="late", u_lo=pscore_lo, u_hi=pscore_hi)]

u_partition = np.unique(np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi_extra, 1]))

bfuncs_const_spline = generate_constant_splines_basis_funcs(u_partition=u_partition)
bfuncs_bernstein = generate_bernstein_basis_funcs(k=k_bernstein)


def _at(u: float) -> bool:
    return u <= pscore_lo


def _c(u: float) -> bool:
    return pscore_lo <= u and pscore_hi > u


def _nt(u: float) -> bool:
    return u >= pscore_hi


# The following MTR functions are decreasing and imply a decreasing MTE function.
# TODO: Think about what solution region this implies. Maybe loop over a couple of
# values.
y1_at = 0.75
y0_at = 0.25

y1_nt = 0.75
y0_nt = 0.6


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


class _Arguments(NamedTuple):
    idestimands: str
    constraint_type_to_arg: dict
    bfunc_type: str
    path_to_data: Annotated[Path, Product]
    num_gridpoints: int = 25


_shape_constr = {"shape_constraint": ("decreasing", "decreasing")}
_mte_monotone = {"mte_monotone": "decreasing"}
_monotone_response = {"monotone_response": "positive"}


ID_TO_KWARGS = {
    f"{bfunc}_{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        bfunc_type=bfunc,
        path_to_data=BLD
        / "data"
        / f"solution_simple_model_{bfunc}_{idestimands}_{constraint}",
        constraint_type_to_arg=constraint,  # type: ignore[arg-type]
    )
    for bfunc in ["constant", "bernstein"]
    for idestimands in ["late", "sharp"]
    for constraint in [_shape_constr, _mte_monotone, _monotone_response]
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_solve_simple_model(
        num_gridpoints: int,
        path_to_data: Annotated[Path, Product],
        constraint_type_to_arg: dict,
        bfunc_type: str,
        idestimands: str,
    ) -> None:
        """Solve the simple model for a range of parameter values."""
        param_grid = np.linspace(0, 1, num_gridpoints)

        # Generate solution for a meshgrid of parameter values
        y1_c, y0_c = np.meshgrid(param_grid, param_grid)

        # Flatten the meshgrid
        y1_c_flat = y1_c.flatten()
        y0_c_flat = y0_c.flatten()

        # Get inputs
        bfuncs = bfuncs_const_spline if bfunc_type == "constant" else bfuncs_bernstein
        identified = identified_late if idestimands == "late" else identified_sharp

        results = []

        for y1_c_val, y0_c_val in zip(y1_c_flat, y0_c_flat, strict=True):
            _m1 = _make_m1(y1_c_val)
            _m0 = _make_m0(y0_c_val)

            # The identified set might be empty for some parameter value combinations.
            res = identification(
                target=target,
                identified_estimands=identified,
                basis_funcs=bfuncs,
                instrument=instrument,
                u_partition=u_partition,
                m0_dgp=_m0,
                m1_dgp=_m1,
                **constraint_type_to_arg,
            )
            res = {"upper_bound": np.nan, "lower_bound": np.nan}

            results.append(res)

        # Put into pandas DataFrame and save to disk

        constraint_type = constraint_type_to_arg.keys()
        constraint_val = constraint_type_to_arg.values()

        df_res = pd.DataFrame(
            {
                "y1_c": y1_c_flat,
                "y0_c": y0_c_flat,
                "upper_bound": [r["upper_bound"] for r in results],
                "lower_bound": [r["lower_bound"] for r in results],
                "y1_at": y1_at,
                "y0_at": y0_at,
                "y1_nt": y1_nt,
                "y0_nt": y0_nt,
            },
        )

        df_res["bfunc_type"] = bfunc_type
        df_res["idestimands"] = idestimands
        df_res["constraint_type"] = constraint_type
        df_res["constraint_val"] = constraint_val

        df_res.to_pickle(path_to_data)
