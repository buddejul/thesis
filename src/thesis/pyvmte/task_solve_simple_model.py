"""Solve the simple model for a specification resulting in sharp bounds."""

import pickle
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
from thesis.pyvmte.pyvmte_sims_config import Y0_AT, Y0_NT, Y1_AT, Y1_NT

# --------------------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------------------
num_gridpoints = 1000

k_bernstein = 11

u_hi_extra = 0.2

shape_constraints = ("decreasing", "decreasing")
mte_monotone = "decreasing"
monotone_response = "positive"

# --------------------------------------------------------------------------------------
# Construct inputs
# --------------------------------------------------------------------------------------


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
y1_at = Y1_AT
y0_at = Y0_AT

y1_nt = Y1_NT
y0_nt = Y0_NT


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
    path_to_dicts: Annotated[Path, Product]
    num_gridpoints: int = num_gridpoints


_none: dict = {}
_shape_constr = {"shape_constraints": shape_constraints}
_mte_monotone = {"mte_monotone": mte_monotone}
_monotone_response = {"monotone_response": monotone_response}


ID_TO_KWARGS = {
    f"{bfunc}_{idestimands}_{constraint}": _Arguments(
        idestimands=idestimands,
        bfunc_type=bfunc,
        constraint_type_to_arg=constraint,  # type: ignore[arg-type]
        path_to_data=BLD
        / "data"
        / "solutions"
        / f"solution_simple_model_{bfunc}_{idestimands}_{constraint}.pkl",
        path_to_dicts=BLD
        / "data"
        / "solutions"
        / f"full_solution_dict_simple_model_{bfunc}_{idestimands}_{constraint}.pkl",
    )
    for bfunc in ["constant", "bernstein"]
    for idestimands in ["late", "sharp"]
    for constraint in [_none, _shape_constr, _mte_monotone, _monotone_response]
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_solve_simple_model(
        num_gridpoints: int,
        constraint_type_to_arg: dict,
        bfunc_type: str,
        idestimands: str,
        path_to_data: Annotated[Path, Product],
        path_to_dicts: Annotated[Path, Product],
    ) -> None:
        """Solve the simple model for a range of parameter values."""
        beta_late = np.linspace(-1, 1, num_gridpoints)

        # Construct y1_c and y0_c such that beta_late = y1_c - y0_c but both are between
        # 0 and 1
        y1_c = beta_late / 2 + 0.5
        y0_c = -beta_late / 2 + 0.5

        # Get inputs
        bfuncs = bfuncs_const_spline if bfunc_type == "constant" else bfuncs_bernstein
        identified = identified_late if idestimands == "late" else identified_sharp

        results = []

        results_full = {}

        for y1_c_val, y0_c_val in zip(y1_c, y0_c, strict=True):
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
            res = {"upper_bound": res.upper_bound, "lower_bound": res.lower_bound}

            results.append(res)

            _b_late = y1_c_val - y0_c_val
            results_full[_b_late] = res

        # Put into pandas DataFrame and save to disk

        if constraint_type_to_arg == {}:
            constraint_type = "none"
            constraint_val = "none"
        else:
            constraint_type = next(iter(constraint_type_to_arg.keys()))
            constraint_val = next(iter(constraint_type_to_arg.values()))

        if isinstance(constraint_val, tuple):
            constraint_val = "_".join(constraint_val)

        df_res = pd.DataFrame(
            {
                "y1_c": y1_c,
                "y0_c": y0_c,
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
        df_res["k_bernstein"] = k_bernstein if bfunc_type == "bernstein" else np.nan
        df_res["num_gridpoints"] = num_gridpoints

        df_res.to_pickle(path_to_data)

        # Save full results in dictionary
        results_full["bfunc_type"] = bfunc_type
        results_full["idestimands"] = idestimands
        results_full["constraint_type"] = constraint_type
        results_full["constraint_val"] = constraint_val
        results_full["k_bernstein"] = (
            k_bernstein if bfunc_type == "bernstein" else np.nan
        )
        results_full["num_gridpoints"] = num_gridpoints

        with Path.open(path_to_dicts, "wb") as f:
            pickle.dump(results_full, f)
