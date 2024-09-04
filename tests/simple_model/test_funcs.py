"""Test functions for simple_model."""

from functools import partial

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG
from thesis.simple_model.funcs import (
    _draw_data,
    _estimate_pscores,
    _late,
    simulation_bootstrap,
)


@pytest.fixture()
def instrument() -> Instrument:
    return Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )


@pytest.fixture()
def local_ates_zero() -> LocalATEs:
    return LocalATEs(
        never_taker=0,
        complier=0,
        always_taker=0,
    )


@pytest.fixture()
def local_ates_nonzero() -> LocalATEs:
    return LocalATEs(
        never_taker=0,
        complier=0.5,
        always_taker=1,
    )


@pytest.fixture()
def sim_boot():
    return partial(
        simulation_bootstrap,
        n_sims=2,
        n_obs=10_000,
        n_boot=2_000,
        u_hi=0.1,
        alpha=0.05,
        rng=RNG,
    )


def test_data_moments_boundary(instrument, local_ates_zero) -> None:
    n_obs = 100_000

    expected = pd.DataFrame(
        {
            "y_given_z": [0.0, 0.0],
            "d_given_z": instrument.pscores,
        },
        index=[0, 1],
    )

    data = pd.DataFrame(
        _draw_data(n_obs, rng=RNG, instrument=instrument, local_ates=local_ates_zero),
        columns=["y", "d", "z", "u"],
    )

    actual = pd.DataFrame(
        {
            "y_given_z": data.groupby("z")["y"].mean(),
            "d_given_z": data.groupby("z")["d"].mean(),
        },
        index=[0, 1],
    )

    pd.testing.assert_frame_equal(actual, expected, atol=0.01)


def test_compute_pscores(instrument, local_ates_nonzero) -> None:
    n_obs = 1_000_000

    data = _draw_data(
        n_obs,
        rng=RNG,
        instrument=instrument,
        local_ates=local_ates_nonzero,
    )

    expected = (0.4, 0.6)

    actual = _estimate_pscores(data)

    assert expected == pytest.approx(actual, abs=3 / np.sqrt(n_obs))


def test_generate_late(instrument, local_ates_nonzero):
    expected = local_ates_nonzero.complier

    data = _draw_data(
        n_obs=100_000,
        local_ates=local_ates_nonzero,
        instrument=instrument,
        rng=RNG,
    )

    actual = _late(data)

    assert actual == pytest.approx(expected, abs=2 * np.sqrt(10_000))


def test_simulation_runs(local_ates_nonzero, instrument, sim_boot) -> None:
    for boot_met in ["standard", "numerical_delta"]:
        for const_mtr in ["increasing", "none"]:
            sim_boot(
                local_ates=local_ates_nonzero,
                instrument=instrument,
                constraint_mtr=const_mtr,
                bootstrap_method=boot_met,
            )


def test_check_invalid_constraint_mtr(instrument, local_ates_nonzero, sim_boot) -> None:
    with pytest.raises(ValueError, match="Constraint 'invalid' not supported."):
        sim_boot(
            local_ates=local_ates_nonzero,
            instrument=instrument,
            constraint_mtr="invalid",
            bootstrap_method="standard",
        )


def test_check_invalid_bootstrap_method(
    instrument,
    local_ates_nonzero,
    sim_boot,
) -> None:
    with pytest.raises(ValueError, match="Bootstrap method 'invalid' not supported."):
        sim_boot(
            local_ates=local_ates_nonzero,
            instrument=instrument,
            constraint_mtr="increasing",
            bootstrap_method="invalid",
        )
