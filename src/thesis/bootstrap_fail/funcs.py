"""Functions for bootstrap sampling experiment."""

from math import comb

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from scipy import integrate  # type: ignore[import-untyped]

PSCORES = [0.4, 0.6]


def simulation_bootstrap(
    n_sims: int,
    n_obs: int,
    n_boot: int,
    u_hi: float,
    alpha: float,
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
        rng: Random number generator.
        param_pos: Position of the parameter in the identified set.

    Returns:
        DataFrame with the results of the simulation.

    """
    results = np.zeros((n_sims, 2))
    targets = np.zeros(n_sims)

    for i in range(n_sims):
        data = _draw_data(n_obs, rng, param_pos=param_pos)
        pscores = _compute_pscores(data)

        results[i] = _bootstrap_ci(alpha, u_hi, data, pscores, n_boot, rng)

        targets[i] = _true_late(u_hi, pscores, param_pos=param_pos)

    out = pd.DataFrame(results, columns=["lo", "hi"])
    out["true"] = targets

    return out


def _bootstrap_ci(
    alpha: float,
    u_hi: float,
    data: np.ndarray,
    pscores: tuple[float, float],
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
        lo, hi = _idset(late, u_hi, pscores)

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
    pscores: tuple[float, float],
) -> tuple[float, float]:
    w = (pscores[1] - pscores[0]) / (u_hi + pscores[1] - pscores[0])

    lo = w * b_late - (1 - w)
    hi = w * b_late + (1 - w)

    return lo, hi


def _late(data: np.ndarray) -> float:
    yz1 = data[data[:, 2] == 1, 0].mean()
    yz0 = data[data[:, 2] == 0, 0].mean()

    dz1 = data[data[:, 2] == 1, 1].mean()
    dz0 = data[data[:, 2] == 0, 1].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def _draw_data(n_obs, rng: np.random.Generator, param_pos: str) -> np.ndarray:
    z = rng.choice([0, 1], size=n_obs, p=[0.5, 0.5])

    u = rng.uniform(low=0, high=1, size=n_obs)

    d = np.where(z == 1, u <= PSCORES[1], u <= PSCORES[0])

    # Situation, where the parameter is at the param_pos of the identified set.
    if param_pos == "boundary":
        y0 = 0
        y1 = np.where(u <= PSCORES[1], 0, 1)
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


def _true_late(u_hi: float, pscores: tuple[float, float], param_pos: str) -> float:
    if param_pos == "boundary":
        return 0 + 1 * (u_hi) / (u_hi + pscores[1] - pscores[0])

    return (
        integrate.quad(_m1, pscores[0], pscores[1] + u_hi)[0]
        - integrate.quad(_m0, pscores[0], pscores[1] + u_hi)[0]
    )


def _compute_pscores(data: np.ndarray) -> tuple[float, float]:
    """Compute the propensity score."""
    return (data[data[:, 2] == 0, 1].mean(), data[data[:, 2] == 1, 1].mean())
