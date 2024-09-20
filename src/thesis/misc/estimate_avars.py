"""Estimate asymptotic variances for some key estimators."""

import numpy as np

from thesis.classes import Instrument, LocalATEs
from thesis.config import RNG
from thesis.simple_model.funcs import (
    _estimate_pscores,
    _late,
)
from thesis.utilities import draw_data

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

n_obs = 1_000_000
n_sim = 10_000
late_complier = 0.5
u_hi = 0.2
rn = np.sqrt(n_obs)

# --------------------------------------------------------------------------------------
# Preliminary calculations
# --------------------------------------------------------------------------------------

local_ates = LocalATEs(
    always_taker=0,
    complier=late_complier,
    never_taker=np.min((1, 1 + late_complier)),
)

instrument = Instrument(
    support=np.array([0, 1]),
    pmf=np.array([0.5, 0.5]),
    pscores=np.array([0.4, 0.6]),
)


true_w = (instrument.pscores[1] - instrument.pscores[0]) / (
    instrument.pscores[1] + u_hi - instrument.pscores[0]
)

true_late = late_complier
true_idset = true_w * late_complier + (1 - true_w) * local_ates.always_taker

# --------------------------------------------------------------------------------------
# Asymptotic variance of LATE, w * LATE, w_hat * LATE and id_set
# --------------------------------------------------------------------------------------

sim_late = np.zeros(n_sim)
sim_late_true_w = np.zeros(n_sim)
sim_late_hat_w = np.zeros(n_sim)
sim_id_set = np.zeros(n_sim)

for i in range(n_sim):
    data = draw_data(
        n_obs=n_obs,
        local_ates=local_ates,
        instrument=instrument,
        rng=RNG,
    )
    _pscores = _estimate_pscores(data)
    hat_w = (_pscores[1] - _pscores[0]) / (_pscores[1] + u_hi - _pscores[0])

    sim_late[i] = _late(data)
    sim_late_true_w[i] = _late(data) * true_w
    sim_late_hat_w[i] = _late(data) * hat_w
    sim_id_set[i] = _late(data) * hat_w + (1 - hat_w)

avar_late = (rn * (sim_late - true_late)).var()
avar_late_true_w = (rn * (sim_late_true_w - true_late * true_w)).var()
avar_late_hat_w = (rn * (sim_late_hat_w - true_late * true_w)).var()
avar_id_set = (rn * (sim_id_set - (true_late * true_w + (1 - true_w)))).var()
