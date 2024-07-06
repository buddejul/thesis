"""Functions for bootstrap sampling experiment."""

from math import comb

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from scipy import integrate  # type: ignore[import-untyped]

from thesis.classes import Instrument


def simulation_bootstrap(
    n_sims: int,
    n_obs: int,
    n_boot: int,
    u_hi: float,
    alpha: float,
    instrument: Instrument,
    rng: np.random.Generator,
    param_pos: str = "boundary",
) -> pd.DataFrame:
    """Simulate the bootstrap experiment.

    Args:
        n_sims: Number of simulations.
        n_obs: Number of observations.
        n_boot: Number of bootstrap samples.
        u_hi: Upper bound of target parameter.
        alpha: Bootstrap percentile for confidence interval (1-alpha for upper).
        instrument: Instrument object containing properties of the instrument.
        rng: Random number generator.
        param_pos: Position of the parameter in the identified set.

    Returns:
        DataFrame with the results of the simulation.

    """
    results = np.zeros((n_sims, 2))

    for i in range(n_sims):
        data = _draw_data(n_obs, rng, param_pos=param_pos, instrument=instrument)

        results[i] = _bootstrap_ci(alpha, u_hi, data, n_boot, rng)

    out = pd.DataFrame(results, columns=["lo", "hi"])
    out["u_hi"] = u_hi
    out["true"] = _true_late(u_hi, instrument=instrument, param_pos=param_pos)

    return out


def _bootstrap_ci(
    alpha: float,
    u_hi: float,
    data: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    return_distr: bool = False,  # noqa: FBT001, FBT002
) -> tuple[ArrayLike, ...]:
    boot_lo = np.zeros(n_boot)
    boot_hi = np.zeros(n_boot)
    boot_late = np.zeros(n_boot)

    n_obs = data.shape[0]

    for i in range(n_boot):
        boot_data = data[rng.choice(n_obs, size=n_obs, replace=True)]
        late = _late(boot_data)
        pscores_hat = _estimate_pscores(boot_data)
        lo, hi = _idset(late, u_hi, pscores_hat)

        boot_late[i] = late
        boot_lo[i] = lo
        boot_hi[i] = hi

    if return_distr:
        return boot_lo, boot_hi, boot_late

    # Take the alpha quantile of boot_lo and 1 - alpha quantile of boot_hi
    return np.quantile(boot_lo, alpha), np.quantile(boot_hi, 1 - alpha)


def _idset(
    b_late: float,
    u_hi: float,
    pscores_hat: tuple[float, float],
) -> tuple[float, float]:
    w = (pscores_hat[1] - pscores_hat[0]) / (u_hi + pscores_hat[1] - pscores_hat[0])

    lo = w * b_late - (1 - w)
    hi = w * b_late + (1 - w)

    return lo, hi


def _late(data: np.ndarray) -> float:
    yz1 = data[data[:, 2] == 1, 0].mean()
    yz0 = data[data[:, 2] == 0, 0].mean()

    dz1 = data[data[:, 2] == 1, 1].mean()
    dz0 = data[data[:, 2] == 0, 1].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def _draw_data(
    n_obs,
    rng: np.random.Generator,
    param_pos: str,
    instrument: Instrument,
) -> np.ndarray:
    z = rng.choice(instrument.support, size=n_obs, p=instrument.pmf)

    u = rng.uniform(low=0, high=1, size=n_obs)

    d = np.where(z == 1, u <= instrument.pscores[1], u <= instrument.pscores[1])

    # Situation, where the parameter is at the boundary of the identified set.
    if param_pos == "boundary":
        y0 = 0
        y1 = np.where(u <= instrument.pscores[1], 0, 1)
        y = d * y1 + (1 - d) * y0 + rng.normal(scale=0.1, size=n_obs)

    return np.column_stack((y, d, z, u))


def _m0(u: float | np.ndarray) -> float | np.ndarray:
    """Compute the m0 function."""
    return 0.6 * _bern(0, 2, u) + 0.4 * _bern(1, 2, u) + 0.3 * _bern(2, 2, u)


def _m1(u: float | np.ndarray) -> float | np.ndarray:
    """Compute the m1 function."""
    return 0.75 * _bern(0, 2, u) + 0.5 * _bern(1, 2, u) + 0.25 * _bern(2, 2, u)


def _bern(v: int, n: int, x: float | np.ndarray) -> float | np.ndarray:
    """Compute the Bernstein polynomial of degree n and index i at x."""
    return comb(n, v) * x**v * (1 - x) ** (n - v)


def _true_late(u_hi: float, instrument: Instrument, param_pos: str) -> float:
    if param_pos == "boundary":
        return 0 + 1 * (u_hi) / (u_hi + instrument.pscores[1] - instrument.pscores[0])

    return (
        integrate.quad(_m1, instrument.pscores[0], instrument.pscores[1] + u_hi)[0]
        - integrate.quad(_m0, instrument.pscores[0], instrument.pscores[1] + u_hi)[0]
    )


def _estimate_pscores(data: np.ndarray) -> tuple[float, float]:
    """Estimate the propensity score."""
    return (data[data[:, 2] == 0, 1].mean(), data[data[:, 2] == 1, 1].mean())
