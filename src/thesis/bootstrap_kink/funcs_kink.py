"""Functions for analyzing inference for simple kink model."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def simulation(n_sim, n_obs, n_boot, theta_0, phi, alpha, rng):
    """Simulation."""
    return pd.concat(
        [
            _experiment(n_obs, n_boot, theta_0, phi, alpha, rng, return_boot=False)
            for _ in range(n_sim)
        ],
        axis=1,
    ).T


def _experiment(n_obs, n_boot, theta_0, phi, alpha, rng, return_boot):
    x = rng.normal(theta_0, 1, n_obs)

    mle = phi(x.mean())

    boot_mle = np.zeros(n_boot)

    for i in range(n_boot):
        boot_idx = rng.choice(n_obs, n_obs, replace=True)
        boot_x = x[boot_idx]
        boot_mle[i] = phi(boot_x.mean())

    boot_distr = np.sqrt(n_obs) * (boot_mle - mle)

    boot_cval_lo = np.percentile(boot_distr, 100 * alpha / 2)
    boot_cval_hi = np.percentile(boot_distr, 100 * (1 - alpha / 2))

    ci_lo = mle - boot_cval_hi / np.sqrt(n_obs)
    ci_hi = mle - boot_cval_lo / np.sqrt(n_obs)

    if return_boot:
        return boot_distr

    return pd.Series(
        {
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "theta_0": theta_0,
            "alpha": alpha,
            "n_obs": n_obs,
            "n_boot": n_boot,
        },
    )


def _phi_max(theta: float, cutoff: float = 0):
    return np.maximum(theta, cutoff)


def _phi_kink(
    theta: float,
    kink: float,
    slope_left: float = 0.5,
    slope_right: float = 1,
):
    return slope_left * theta * (theta < kink) + slope_right * theta * (theta >= kink)
