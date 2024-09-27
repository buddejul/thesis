"""Test solutions to simple model in case for sharp identified set."""

import numpy as np
import pytest
from pyvmte.classes import Estimand, Instrument  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]

atol = 1e-05

# --------------------------------------------------------------------------------------
# Preliminary settings
# --------------------------------------------------------------------------------------
num_gridpoints = 5

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


def _at(u: float) -> bool:
    return u <= pscore_lo


def _c(u: float) -> bool:
    return pscore_lo <= u and pscore_hi > u


def _nt(u: float) -> bool:
    return u >= pscore_hi


# Define function factories to avoid late binding
# See https://stackoverflow.com/a/3431699
def _make_m0(y0_c, y0_at, y0_nt):
    def _m0(u):
        return y0_at * _at(u) + y0_c * _c(u) + y0_nt * _nt(u)

    return _m0


def _make_m1(y1_c, y1_at, y1_nt):
    def _m1(u):
        return y1_at * _at(u) + y1_c * _c(u) + y1_nt * _nt(u)

    return _m1


def _sol_hi_ate(w, y1_c, y0_c, y0_nt):
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (y1_c - y0_nt)


def _sol_lo_ate(w, y1_c, y0_c, y0_nt):
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (0 - y0_nt)


def _sol_hi_late(w, y1_c, y0_c, y0_nt):
    return _sol_hi_ate(w, y1_c, y0_c, y0_nt)


def _sol_lo_late(u_hi, w, pscore_hi, y1_c, y0_c, y0_nt):
    _b_late = y1_c - y0_c

    k = u_hi / (1 - pscore_hi)

    return w * _b_late + (1 - w) * (0 - np.minimum(y0_c, y0_nt / k))


def _no_solution(y1_at, y1_c, y0_c, y0_nt):
    # The model is not consistent with decreasing MTRs if
    # - y1_at < y1_c or y1_c < y1_nt or
    # - y0_at < y0_c or y0_c < y0_nt
    # Make sure this also works vectorized.
    # Note that while y1_c < y1_nt is also not consistent with decreasing MTRs, the
    # model puts no restrictions on y1_nt since it is not identified.
    return np.logical_or(y1_at < y1_c, y0_c < y0_nt)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@pytest.mark.skip()
def test_solve_simple_model_sharp_ate_decreasing() -> None:
    """Solve the simple model for a range of parameter values."""
    u_hi = 1 - pscore_hi

    target = Estimand(
        "late",
        u_lo=pscore_lo,
        u_hi=pscore_hi + u_hi,
    )

    bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": pscore_lo}
    bfunc_2 = {"type": "constant", "u_lo": pscore_lo, "u_hi": pscore_hi}
    bfunc_3 = {"type": "constant", "u_lo": pscore_hi, "u_hi": pscore_hi + u_hi}

    bfuncs = [bfunc_1, bfunc_2, bfunc_3]

    u_partition = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi])

    _grid = np.linspace(0, 1, num_gridpoints)

    w = (pscore_hi - pscore_lo) / (pscore_hi - pscore_lo + u_hi)

    shape_constraint = ("decreasing", "decreasing")

    # Generate solution for a meshgrid of parameter values
    (
        y1_at,
        y1_c,
        y1_nt,
        y0_at,
        y0_c,
        y0_nt,
    ) = np.meshgrid(_grid, _grid, _grid, _grid, _grid, _grid)

    # Flatten each meshgrid
    y1_at_flat = y1_at.flatten()
    y1_c_flat = y1_c.flatten()
    y1_nt_flat = y1_nt.flatten()
    y0_at_flat = y0_at.flatten()
    y0_c_flat = y0_c.flatten()
    y0_nt_flat = y0_nt.flatten()

    results = []

    for y1_at, y1_c, y1_nt, y0_at, y0_c, y0_nt in zip(
        y1_at_flat,
        y1_c_flat,
        y1_nt_flat,
        y0_at_flat,
        y0_c_flat,
        y0_nt_flat,
        strict=True,
    ):
        _m1 = _make_m1(y1_at=y1_at, y1_c=y1_c, y1_nt=y1_nt)
        _m0 = _make_m0(y0_at=y0_at, y0_c=y0_c, y0_nt=y0_nt)

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

    actual_hi = np.array([res["upper_bound"] for res in results])
    actual_lo = np.array([res["lower_bound"] for res in results])

    # Put into pandas DataFrame and save to disk
    _kwargs = {
        "y1_c": y1_c_flat,
        "y0_c": y0_c_flat,
        "y0_nt": y0_nt_flat,
    }

    expected_hi = _sol_hi_ate(w=w, **_kwargs)
    expected_lo = _sol_lo_ate(w=w, **_kwargs)

    _idx_no_sol = _no_solution(y1_at=y1_at_flat, **_kwargs)
    expected_hi[_idx_no_sol] = np.nan
    expected_lo[_idx_no_sol] = np.nan

    # Get _idx of nan mismatch
    np.where(np.isnan(actual_hi) != np.isnan(expected_hi))
    np.where(np.isnan(actual_lo) != np.isnan(expected_lo))

    # Get _idx of value mismatch
    np.where(np.abs(actual_hi - expected_hi) > atol)
    np.where(np.abs(actual_lo - expected_lo) > atol)

    np.testing.assert_allclose(actual_hi, expected_hi, atol=atol)
    np.testing.assert_allclose(actual_lo, expected_lo, atol=atol)


def test_solve_simple_model_sharp_late_decreasing() -> None:
    """Solve the simple model for a range of parameter values."""
    u_hi = 0.2

    target = Estimand(
        "late",
        u_lo=pscore_lo,
        u_hi=pscore_hi + u_hi,
    )

    bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": pscore_lo}
    bfunc_2 = {"type": "constant", "u_lo": pscore_lo, "u_hi": pscore_hi}
    bfunc_3 = {"type": "constant", "u_lo": pscore_hi, "u_hi": pscore_hi + u_hi}
    bfunc_4 = {"type": "constant", "u_lo": pscore_hi + u_hi, "u_hi": 1}

    bfuncs = [bfunc_1, bfunc_2, bfunc_3, bfunc_4]

    u_partition = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi, 1])

    _grid = np.linspace(0, 1, num_gridpoints)

    w = (pscore_hi - pscore_lo) / (pscore_hi - pscore_lo + u_hi)

    shape_constraint = ("decreasing", "decreasing")

    # Generate solution for a meshgrid of parameter values
    (
        y1_at,
        y1_c,
        y1_nt,
        y0_at,
        y0_c,
        y0_nt,
    ) = np.meshgrid(_grid, _grid, _grid, _grid, _grid, _grid)

    # Flatten each meshgrid
    y1_at_flat = y1_at.flatten()
    y1_c_flat = y1_c.flatten()
    y1_nt_flat = y1_nt.flatten()
    y0_at_flat = y0_at.flatten()
    y0_c_flat = y0_c.flatten()
    y0_nt_flat = y0_nt.flatten()

    results = []

    for y1_at, y1_c, y1_nt, y0_at, y0_c, y0_nt in zip(
        y1_at_flat,
        y1_c_flat,
        y1_nt_flat,
        y0_at_flat,
        y0_c_flat,
        y0_nt_flat,
        strict=True,
    ):
        _m1 = _make_m1(y1_at=y1_at, y1_c=y1_c, y1_nt=y1_nt)
        _m0 = _make_m0(y0_at=y0_at, y0_c=y0_c, y0_nt=y0_nt)

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

    actual_hi = np.array([res["upper_bound"] for res in results])
    actual_lo = np.array([res["lower_bound"] for res in results])

    # Put into pandas DataFrame and save to disk
    _kwargs = {
        "y1_c": y1_c_flat,
        "y0_c": y0_c_flat,
        "y0_nt": y0_nt_flat,
    }

    expected_hi = _sol_hi_late(w=w, **_kwargs)
    expected_lo = _sol_lo_late(w=w, pscore_hi=pscore_hi, u_hi=u_hi, **_kwargs)

    _idx_no_sol = _no_solution(y1_at=y1_at_flat, **_kwargs)
    expected_hi[_idx_no_sol] = np.nan
    expected_lo[_idx_no_sol] = np.nan

    # Get _idx of nan mismatch
    np.where(np.isnan(actual_hi) != np.isnan(expected_hi))
    np.where(np.isnan(actual_lo) != np.isnan(expected_lo))

    # Get _idx of value mismatch
    np.where(np.abs(actual_hi - expected_hi) > atol)
    np.where(np.abs(actual_lo - expected_lo) > atol)

    np.testing.assert_allclose(actual_hi, expected_hi, atol=atol)
    np.testing.assert_allclose(actual_lo, expected_lo, atol=atol)
