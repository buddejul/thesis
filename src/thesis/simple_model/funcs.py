"""Functions for bootstrap sampling experiment."""

from math import comb

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from thesis.classes import Instrument, LocalATEs


def simulation_bootstrap(
    n_sims: int,
    n_obs: int,
    n_boot: int,
    u_hi: float,
    local_ates: LocalATEs,
    instrument: Instrument,
    alpha: float,
    constraint_mtr: str,
    bootstrap_method: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate the bootstrap experiment.

    Args:
        n_sims: Number of simulations.
        n_obs: Number of observations.
        n_boot: Number of bootstrap samples.
        local_ates: Local average treatment effects by complier type.
        u_hi: Upper bound of target parameter.
        instrument: Instrument object containing properties of the instrument.
        alpha: Bootstrap percentile for confidence interval (1-alpha for upper).
        constraint_mtr: Constraint on the marginal treatment response functions.
        rng: Random number generator.
        bootstrap_method: Method to compute the bootstrap confidence interval.

    Returns:
        DataFrame with the results of the simulation.

    """
    _check_constraint_supported(constraint_mtr)
    _check_bootsrap_method_supported(bootstrap_method)

    results = np.zeros((n_sims, 2))

    for i in range(n_sims):
        data = _draw_data(n_obs, local_ates=local_ates, instrument=instrument, rng=rng)

        results[i] = _bootstrap_ci(
            n_boot=n_boot,
            bootstrap_method=bootstrap_method,
            alpha=alpha,
            u_hi=u_hi,
            data=data,
            constraint_mtr=constraint_mtr,
            rng=rng,
        )

    out = pd.DataFrame(results, columns=["lo", "hi"])
    out["true"] = _true_late(u_hi, instrument=instrument, local_ates=local_ates)

    return out


def _bootstrap_ci(
    bootstrap_method: str,
    alpha: float,
    u_hi: float,
    data: np.ndarray,
    n_boot: int,
    constraint_mtr: str,
    rng: np.random.Generator,
) -> tuple[ArrayLike, ...]:
    """Compute bootstrap confidence interval.

    The function allows for different type of bootstrap confidence intervals:
    - standard: Calculates phi(beta_s) for every bootstrap drawn. Then it returns the
        quantiles of the bootstrap distribution.
    - numerical delta: Separately bootstraps the distribution of beta_s and estimates a
        a derivative phi'(beta_s) using a numerical approximation.
        Based on Hong and Li (2018).
    - analytical delta: Separately bootstraps the distribution of beta_s and estimates a
        a derivative phi'(beta_s) utilizing the analytical structure of the derivative.
        Based on Fang and Santos (2017).

    """
    # Note that we currently implement this by separately bootstrapping the data for
    # each method. This is not the most efficient way to do this, but keeps the code
    # easy to organize. If it turns we out we spend a significant time on resampling
    # we might want to consider a more efficient implementation.

    if bootstrap_method == "standard":
        return _ci_standard_bootstrap(
            n_boot=n_boot,
            data=data,
            alpha=alpha,
            u_hi=u_hi,
            constraint_mtr=constraint_mtr,
            rng=rng,
        )
    # In principle, this should not be needed because we check the supported methods
    # after input. However, mypy does not seem to recognize this. Or maybe this function
    # is called from somewhere else.
    msg = f"Bootstrap method '{bootstrap_method}' not supported."
    raise ValueError(msg)


def _ci_standard_bootstrap(
    n_boot: int,
    data: np.ndarray,
    alpha: float,
    u_hi: float,
    constraint_mtr: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute the standard bootstrap confidence interval.

    The standard, non-parametric approach calculates phi(X_hat) for each bootstrap. In
    our case this amounts to computing the identified set based on the bootstrap sample.
    Then quantiles of this bootstrap distribution are used to construct CIs.

    """
    boot_lo = np.zeros(n_boot)
    boot_hi = np.zeros(n_boot)

    n_obs = data.shape[0]

    for i in range(n_boot):
        boot_data = data[rng.choice(n_obs, size=n_obs, replace=True)]

        boot_lo[i], boot_hi[i] = _idset(
            b_late=_late(boot_data),
            u_hi=u_hi,
            pscores_hat=_estimate_pscores(boot_data),
            constraint_mtr=constraint_mtr,
        )

    # Explicitly make this a float to avoid static typing issues. np.quantile returns a
    # np.float64 type, which is not compatible with the float type hint, although this
    # seems to be addressed in https://github.com/numpy/numpy/pull/27334.
    return float(np.quantile(boot_lo, alpha)), float(np.quantile(boot_hi, 1 - alpha))


def _idset(
    b_late: float,
    u_hi: float,
    pscores_hat: tuple[float, float],
    constraint_mtr: str,
) -> tuple[float, float]:
    w = (pscores_hat[1] - pscores_hat[0]) / (u_hi + pscores_hat[1] - pscores_hat[0])

    if constraint_mtr == "none":
        lo = w * b_late - (1 - w)
        hi = w * b_late + (1 - w)

    # Restricting the MTRs to be increasing changes the solution to have a kink when
    # viewed as a function of the identified parameter.
    elif constraint_mtr == "increasing":
        hi = _phi_kink(theta=b_late, kink=0, slope_left=1, slope_right=w) + (1 - w)
        lo = _phi_kink(theta=b_late, kink=0, slope_left=w, slope_right=1) - (1 - w)

    return lo, hi


def _late(data: np.ndarray) -> float:
    yz1 = data[data[:, 2] == 1, 0].mean()
    yz0 = data[data[:, 2] == 0, 0].mean()

    dz1 = data[data[:, 2] == 1, 1].mean()
    dz0 = data[data[:, 2] == 0, 1].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def _draw_data(
    n_obs,
    local_ates: LocalATEs,
    instrument: Instrument,
    rng: np.random.Generator,
) -> np.ndarray:
    z = rng.choice(instrument.support, size=n_obs, p=instrument.pmf)

    u = rng.uniform(low=0, high=1, size=n_obs)

    d = np.where(z == 1, u <= instrument.pscores[1], u <= instrument.pscores[0])

    _never_takers = u <= instrument.pscores[0]
    _compliers = (u > instrument.pscores[0]) & (u <= instrument.pscores[1])
    _always_takers = u > instrument.pscores[1]

    y0 = np.zeros(n_obs)

    y1 = (
        _never_takers * local_ates.never_taker
        + _compliers * local_ates.complier
        + _always_takers * local_ates.always_taker
    )

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


def _true_late(u_hi: float, instrument: Instrument, local_ates: LocalATEs) -> float:
    _target_pop_size = u_hi + instrument.pscores[1] - instrument.pscores[0]
    _complier_share = (instrument.pscores[1] - instrument.pscores[0]) / _target_pop_size

    return (
        _complier_share * local_ates.complier
        + (1 - _complier_share) * local_ates.always_taker
    )


def _estimate_pscores(data: np.ndarray) -> tuple[float, float]:
    """Estimate the propensity score."""
    return (data[data[:, 2] == 0, 1].mean(), data[data[:, 2] == 1, 1].mean())


def _phi_max(theta: float, cutoff: float = 0):
    return np.maximum(theta, cutoff)


def _phi_kink(
    theta: float,
    kink: float,
    slope_left: float = 0.5,
    slope_right: float = 1,
):
    return slope_left * theta * (theta < kink) + slope_right * theta * (theta >= kink)


def _check_constraint_supported(constraint_mtr: str) -> None:
    supported = ["none", "increasing"]
    if constraint_mtr not in supported:
        msg = (
            f"Constraint '{constraint_mtr}' not supported.\n"
            f"Supported constraints: {supported}"
        )
        raise ValueError(msg)


def _check_bootsrap_method_supported(bootstrap_method: str) -> None:
    supported = ["standard"]
    if bootstrap_method not in supported:
        msg = (
            f"Bootstrap method '{bootstrap_method}' not supported.\n"
            f"Supported constraints: {supported}"
        )
        raise ValueError(msg)
