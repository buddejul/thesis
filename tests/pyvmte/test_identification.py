"""Test identification for simple model using pyvmte."""

import numpy as np
import pytest
from pyvmte.classes import Estimand, Instrument  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
)


@pytest.fixture()
def setup() -> dict:
    u_hi = 0.2

    target = Estimand(
        "late",
        u_lo=0.4,
        u_hi=0.6 + u_hi,
    )

    identified = Estimand(
        "late",
        u_lo=0.4,
        u_hi=0.6,
    )

    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": 0.4}
    bfunc_2 = {"type": "constant", "u_lo": 0.4, "u_hi": 0.6}
    bfunc_3 = {"type": "constant", "u_lo": 0.6, "u_hi": 0.8}
    bfunc_4 = {"type": "constant", "u_lo": 0.8, "u_hi": 1.0}

    basis_funcs_constant = [bfunc_1, bfunc_2, bfunc_3, bfunc_4]

    basis_funcs_bern = generate_bernstein_basis_funcs(k=10)

    u_partition = np.array([0, 0.4, 0.6, 0.8, 1])

    return {
        "target": target,
        "identified_estimands": identified,
        "instrument": instrument,
        "basis_funcs_constant": basis_funcs_constant,
        "basis_funcs_bern": basis_funcs_bern,
        "u_partition": u_partition,
    }


@pytest.mark.skip()
def test_pyvmte_identification_simple_model(setup: dict) -> None:
    num_grid = 100

    late_complier_grid = np.linspace(-1, 1, num_grid)

    expected_hi = 0.5 * late_complier_grid + 0.5 * 1
    expected_lo = 0.5 * late_complier_grid + 0.5 * (-1)
    actual_hi = np.zeros(num_grid)
    actual_lo = np.zeros(num_grid)

    _p1 = setup["instrument"].pscores[1]
    _p0 = setup["instrument"].pscores[0]

    for i in range(num_grid):

        def _m0(u: float) -> float:
            return 0 * u

        def _m1(u, i=i):
            return late_complier_grid[i] * (u >= _p0 and u < _p1)

        _res = identification(
            target=setup["target"],
            identified_estimands=setup["identified_estimands"],
            instrument=setup["instrument"],
            basis_funcs=setup["basis_funcs_constant"],
            u_partition=setup["u_partition"],
            m0_dgp=_m0,
            m1_dgp=_m1,
        )

        actual_hi[i] = _res["upper_bound"]
        actual_lo[i] = _res["lower_bound"]

    assert np.allclose(expected_hi, actual_hi, atol=1e-2)
    assert np.allclose(expected_lo, actual_lo, atol=1e-2)


@pytest.mark.skip()
def test_pyvmte_identification_simple_model_increasing_mtr(setup: dict) -> None:
    num_grid = 100

    late_complier = np.linspace(-1, 1, num_grid)

    w = 0.5

    expected_hi = np.where(
        late_complier >= 0,
        w * late_complier + (1 - w) * 1,
        late_complier + (1 - w) * 1,
    )

    expected_lo = np.where(
        late_complier >= 0,
        late_complier - (1 - w) * 1,
        w * late_complier - (1 - w) * 1,
    )

    actual_hi = np.zeros(num_grid)
    actual_lo = np.zeros(num_grid)

    _p1 = setup["instrument"].pscores[1]
    _p0 = setup["instrument"].pscores[0]

    for i in range(num_grid):

        def _m0(u: float) -> float:
            return u * 0

        def _m1(u, i=i):
            return late_complier[i] * (u >= _p0 and u < _p1)

        _res = identification(
            target=setup["target"],
            identified_estimands=setup["identified_estimands"],
            instrument=setup["instrument"],
            basis_funcs=setup["basis_funcs_constant"],
            u_partition=setup["u_partition"],
            m0_dgp=_m0,
            m1_dgp=_m1,
            shape_constraints=("increasing", "increasing"),
        )

        actual_hi[i] = _res["upper_bound"]
        actual_lo[i] = _res["lower_bound"]

    assert np.allclose(expected_hi, actual_hi, atol=1e-2)
    assert np.allclose(expected_lo, actual_lo, atol=1e-2)
