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
    _idset,
    _late,
    _late_2sls,
    _true_late,
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


# Setting where true parameter is at upper boundary of ID set and at kink of solution
@pytest.fixture()
def local_ates_boundary_hi() -> LocalATEs:
    return LocalATEs(
        never_taker=0,
        complier=0,
        always_taker=1,
    )


# Setting where true parameter is at lower boundary of ID set and at kink of solution
@pytest.fixture()
def local_ates_boundary_lo() -> LocalATEs:
    return LocalATEs(
        never_taker=0,
        complier=0,
        always_taker=-1,
    )


@pytest.fixture()
def local_ates_complier_negative() -> LocalATEs:
    return LocalATEs(
        never_taker=0,
        complier=-0.5,
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

    assert actual == pytest.approx(expected, abs=2 / np.sqrt(10_000))


def test_simulation_runs(local_ates_boundary_hi, instrument, sim_boot) -> None:
    for boot_met in ["analytical_delta", "standard", "numerical_delta"]:
        for const_mtr in ["increasing", "none"]:
            if boot_met == "numerical_delta":
                bootstrap_params = {"eps_fun": lambda n: n ** (-1 / 3)}
            elif boot_met == "analytical_delta":
                bootstrap_params = {"kappa_fun": lambda n: n ** (1 / 3)}
            else:
                bootstrap_params = {}

            sim_boot(
                local_ates=local_ates_boundary_hi,
                instrument=instrument,
                constraint_mtr=const_mtr,
                bootstrap_method=boot_met,
                bootstrap_params=bootstrap_params,
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


def test_true_late():
    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    u_hi = 0.2

    complier_lates = np.linspace(-0.1, 0.1, num=10)

    # With u_hi = 0.2, the weights are 0.5 (both relevant populations have mass 0.2).
    expected = 0.5 * complier_lates + 0.5 * 1

    local_ates = [
        LocalATEs(
            never_taker=0,
            complier=cp_late,
            always_taker=1,
        )
        for cp_late in complier_lates
    ]

    actual = [
        _true_late(instrument=instrument, local_ates=la, u_hi=u_hi) for la in local_ates
    ]

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_id_set_consistent() -> None:
    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    u_hi = 0.2

    local_ates = LocalATEs(
        never_taker=0,
        complier=-0.1,
        always_taker=1,
    )

    constraint_mtr = "none"

    n_obs = 10_000
    n_sims = 1_000

    res = np.zeros((n_sims, 2))

    for i in range(n_sims):
        data = _draw_data(
            n_obs=n_obs,
            local_ates=local_ates,
            instrument=instrument,
            rng=RNG,
        )

        res[i] = _idset(
            b_late=_late(data),
            u_hi=u_hi,
            pscores_hat=_estimate_pscores(data),
            constraint_mtr=constraint_mtr,
        )

    # Take means over columns

    actual = res.mean(axis=0)

    actual[0]
    mean_hi = actual[1]

    expected_hi = 0.45
    assert mean_hi == pytest.approx(expected_hi, abs=3 / np.sqrt(n_obs))


def test_inconsistent_complier_and_always_taker_ate(
    sim_boot,
    local_ates_complier_negative,
    instrument,
) -> None:
    # In settings with no constraint on the MTR functions it is possible to have a
    # negative complier ATE and a always-taker ATE of 1.
    sim_boot(
        local_ates=local_ates_complier_negative,
        instrument=instrument,
        constraint_mtr="none",
        bootstrap_method="standard",
    )

    # With increasing MTR functions this is not possible, as the largest possible value
    # given a negative complier ATE is 1 + complier ATE < 1.

    with pytest.raises(ValueError, match="largest possible always-taker ATE"):
        sim_boot(
            local_ates=local_ates_complier_negative,
            instrument=instrument,
            constraint_mtr="increasing",
            bootstrap_method="standard",
        )


def test_bootstrap_params_supplied(sim_boot, local_ates_nonzero, instrument) -> None:
    with pytest.raises(ValueError, match="Numerical delta bootstrap method requires"):
        sim_boot(
            local_ates=local_ates_nonzero,
            instrument=instrument,
            constraint_mtr="increasing",
            bootstrap_method="numerical_delta",
            bootstrap_params={},
        )
    with pytest.raises(ValueError, match="Analytical delta bootstrap method requires"):
        sim_boot(
            local_ates=local_ates_nonzero,
            instrument=instrument,
            constraint_mtr="increasing",
            bootstrap_method="analytical_delta",
            bootstrap_params={},
        )


def test_late_and_late_2sls_equivalent(local_ates_nonzero, instrument) -> None:
    data = _draw_data(
        n_obs=10_000,
        local_ates=local_ates_nonzero,
        instrument=instrument,
        rng=RNG,
    )

    late = _late(data)

    late_2sls, _ = _late_2sls(data)

    assert late == pytest.approx(late_2sls)
