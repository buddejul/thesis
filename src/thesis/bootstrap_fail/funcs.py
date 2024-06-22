"""Functions for bootstrap sampling experiment."""

from math import comb

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike


def simulation_bootstrap(
    n_sims: int,
    n_obs: int,
    n_boot: int,
    u_hi: float,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate the bootstrap experiment.

    Args:
        n_sims: Number of simulations.
        n_obs: Number of observations.
        n_boot: Number of bootstrap samples.
        u_hi: Upper bound of target parameter.
        alpha: Bootstrap percentile for confidence interval (1-alpha for upper).
        rng: Random number generator.

    Returns:
        DataFrame with the results of the simulation.

    """
    results = np.zeros((n_sims, 2))

    for i in range(n_sims):
        data = _draw_data(n_obs, rng)
        lo, hi = _bootstrap_ci(alpha, u_hi, data, n_boot, n_obs, rng)
        results[i] = lo, hi

    return pd.DataFrame(results, columns=["lo", "hi"])


def _bootstrap_ci(
    alpha: float,
    u_hi: float,
    data: np.ndarray,
    n_boot: int,
    n_obs: int,
    rng: np.random.Generator,
) -> tuple[ArrayLike, ArrayLike]:
    boot_lo = np.zeros(n_boot)
    boot_hi = np.zeros(n_boot)

    for i in range(n_boot):
        boot_data = data[rng.choice(n_obs, size=n_obs, replace=True)]
        late = _late(boot_data)
        lo, hi = _idset(late, u_hi)

        boot_lo[i] = lo
        boot_hi[i] = hi

    # Take the alpha quantile of boot_lo and 1 - alpha quantile of boot_hi
    return np.quantile(boot_lo, alpha), np.quantile(boot_hi, 1 - alpha)


def _idset(b_late: float, u_hi: float) -> tuple[float, float]:
    lo = (0.6 - 0.4) / (u_hi - 0.4) * b_late - (u_hi - 0.6) / (u_hi - 0.4)
    hi = (0.6 - 0.4) / (u_hi - 0.4) * b_late + (u_hi - 0.6) / (u_hi - 0.4)

    return lo, hi


def _late(data: np.ndarray) -> float:
    yz1 = data[data[:, 2] == 1, 0].mean()
    yz0 = data[data[:, 2] == 0, 0].mean()

    dz1 = data[data[:, 2] == 1, 1].mean()
    dz0 = data[data[:, 2] == 0, 1].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def _draw_data(n_obs, rng: np.random.Generator) -> np.ndarray:
    z = rng.choice([0, 1], size=n_obs, p=[0.5, 0.5])

    d = rng.choice([0, 1], size=n_obs, p=[0.4, 0.6]) * z + rng.choice(
        [0, 1],
        size=n_obs,
        p=[0.6, 0.4],
    ) * (1 - z)

    u = rng.uniform(low=0, high=1, size=n_obs)

    y = d * _m1(u) + (1 - d) * _m0(u)

    return np.column_stack((y, d, z))


def _m0(u: float | np.ndarray) -> float | np.ndarray:
    """Compute the m0 function."""
    return 0.6 * _bern(0, 2, u) + 0.4 * _bern(1, 2, u) + 0.3 * _bern(2, 2, u)


def _m1(u: float | np.ndarray) -> float | np.ndarray:
    """Compute the m1 function."""
    return 0.75 * _bern(0, 2, u) + 0.5 * _bern(1, 2, u) + 0.25 * _bern(2, 2, u)


def _bern(v: int, n: int, x: float | np.ndarray) -> float | np.ndarray:
    """Compute the Bernstein polynomial of degree n and index i at x."""
    return comb(n, v) * x**v * (1 - x) ** (n - v)
