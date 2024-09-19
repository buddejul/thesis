"""Test bootstrap confidence intervals have right coverage far off the kink."""

import numpy as np
import pytest
from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG
from thesis.simple_model.funcs import simulation_bootstrap


@pytest.fixture()
def setup():
    complier_late = 0.2

    local_ates = LocalATEs(
        never_taker=0,
        complier=complier_late,
        always_taker=np.min((1, 1 + complier_late)),
    )

    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    true = 0.5 * complier_late + 0.5

    return local_ates, instrument, complier_late, true


def _bic(n):
    """BIC."""
    return np.sqrt(np.log(n))


def test_bootstrap_coverage(setup):
    local_ates, instrument, complier_late, true = setup

    expected = 0.95

    res = simulation_bootstrap(
        n_sims=500,
        n_obs=10_000,
        n_boot=10_000,
        u_hi=0.2,
        local_ates=local_ates,
        instrument=instrument,
        alpha=0.05,
        constraint_mtr="increasing",
        bootstrap_method="analytical_delta",
        rng=RNG,
        bootstrap_params={"kappa_fun": _bic},
    )

    res["covers"] = (res["lo"] <= true) & (res["hi"] >= true)

    actual = res.mean()["covers"]

    assert actual == pytest.approx(expected, abs=0.025)
