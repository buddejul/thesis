"""Tests for debugging simple model CIs."""

import numpy as np
from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG
from thesis.simple_model.funcs import simulation_bootstrap


def test_debug_sm():
    n_sims = 2
    n_obs = 10_000
    n_boot = 2_000
    u_hi = 0.2
    alpha = 0.05
    constraint_mtr = "none"
    bootstrap_method = "standard"

    instrument = Instrument(
        support=np.array([0, 1]),
        pmf=np.array([0.5, 0.5]),
        pscores=np.array([0.4, 0.6]),
    )

    local_ates = LocalATEs(
        never_taker=0,
        complier=-0.1,
        always_taker=1,
    )

    simulation_bootstrap(
        n_sims=n_sims,
        n_obs=n_obs,
        n_boot=n_boot,
        u_hi=u_hi,
        alpha=alpha,
        rng=RNG,
        constraint_mtr=constraint_mtr,
        bootstrap_method=bootstrap_method,
        instrument=instrument,
        local_ates=local_ates,
    )
