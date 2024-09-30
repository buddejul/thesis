"""Test bootstrap confidence intervals have right coverage far off the kink."""

import numpy as np
import pytest
from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG
from thesis.simple_model.funcs import simulation_bootstrap
<<<<<<< HEAD
from thesis.utilities import bic
=======
>>>>>>> main


@pytest.fixture()
def setup():
    complier_late = -0.5

    local_ates = LocalATEs(
        always_taker=0,
        complier=complier_late,
        never_taker=np.min((1, 1 + complier_late)),
    )

    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    return local_ates, instrument, complier_late


<<<<<<< HEAD
=======
def _bic(n):
    """BIC."""
    return np.sqrt(np.log(n))


>>>>>>> main
@pytest.mark.parametrize("method", ["numerical_delta"])
def test_bootstrap_coverage(setup, method):
    local_ates, instrument, complier_late = setup

    expected = 0.95

    n_obs = 10_000
    n_boot = 500
    n_sims = 100

    bootstrap_params = {
        "eps_fun": np.sqrt,
<<<<<<< HEAD
        "kappa_fun": bic,
=======
        "kappa_fun": _bic,
>>>>>>> main
    }

    res = simulation_bootstrap(
        n_sims=n_sims,
        n_obs=n_obs,
        n_boot=n_boot,
        u_hi=0.2,
        local_ates=local_ates,
        instrument=instrument,
        alpha=0.05,
        constraint_mtr="increasing",
        bootstrap_method=method,
        rng=RNG,
        bootstrap_params=bootstrap_params,
    )

    res["covers"] = (res["lo"] <= res["true"]) & (res["hi"] >= res["true"])
    res["covers_hi"] = res["hi"] >= res["true"]
    res["covers_lo"] = res["lo"] <= res["true"]

    # Calculate critical values of CI
    res["c_hi"] = res["hi"] - res["beta_hi"]
    res["c_lo"] = res["lo"] - res["beta_lo"]

    # Assert they are always >= 0
    assert np.all(res["c_hi"] >= 0)
    assert np.all(res["c_lo"] <= 0)

    # Coverage should be determined by upper bound
    assert res["covers_lo"].mean() == 1
    assert np.all(res["covers_hi"] == res["covers"])

    actual = res.mean()["covers"]

    assert actual == pytest.approx(expected, abs=0.051)
