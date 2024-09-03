"""Test functions for bootstrap_fail."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from thesis.bootstrap_fail.funcs import _draw_data, _estimate_pscores, _late
from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG


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
