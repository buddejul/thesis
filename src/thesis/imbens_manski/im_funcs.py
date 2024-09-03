"""Functions for Imbens and Manski (2004) ECMA simulations."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]

from thesis.config import RNG


def simulation(
    n_sims: int,
    n_obs: int,
    p: float,
    alpha: float,
    ci_type: str,
    rng: np.random.Generator = RNG,
    beta_params: tuple[float, float] = (1, 1),
    n_boot: int = 1_000,
) -> pd.DataFrame:
    """Run simulation."""
    columns = [
        "cilo_idset",
        "cihi_idset",
        "cilo_param",
        "cihi_param",
        "mu_bar",
        "sigma_sq_bar",
    ]

    res = pd.DataFrame(
        [
            _experiment(
                n_obs,
                p,
                alpha,
                beta_params=beta_params,
                rng=rng,
                ci_type=ci_type,
                n_boot=n_boot,
            )
            for _ in range(n_sims)
        ],
        columns=columns,
    )

    mean = beta_params[0] / (beta_params[0] + beta_params[1])

    res["ci_idset_covers"] = (res["cilo_idset"] <= mean) & (mean <= res["cihi_idset"])
    res["ci_param_covers"] = (res["cilo_param"] <= mean) & (mean <= res["cihi_param"])

    return res


def _experiment(
    n_obs,
    p: float,
    alpha: float,
    beta_params: tuple[float, float],
    ci_type: str,
    n_boot: int = 1_000,
    rng: np.random.Generator = RNG,
):
    y, w = _draw_data(n_obs, p, beta_params=beta_params, rng=rng)

    n1 = np.sum(w)

    if n1 == 0:
        return 0, 1

    mu_bar = np.sum(y * w) / n1
    sigma_sq_bar = np.sum(w * (y - mu_bar) ** 2) / np.clip(
        (n1 - 1),
        a_min=1,
        a_max=None,
    )

    if ci_type == "analytical":
        cilo_idset, cihi_idset = _compute_ci(
            alpha,
            p,
            mu_bar,
            sigma_sq_bar,
            n_obs,
            target="idset",
        )
        cilo_param, cihi_param = _compute_ci(
            alpha,
            p,
            mu_bar,
            sigma_sq_bar,
            n_obs,
            target="param",
        )

    elif ci_type == "bootstrap":
        cilo_idset, cihi_idset = _compute_bootstrap_ci(
            n_boot=n_boot,
            y=y,
            w=w,
            alpha=alpha,
            p=p,
            target="idset",
        )
        cilo_param, cihi_param = _compute_bootstrap_ci(
            n_boot=n_boot,
            y=y,
            w=w,
            alpha=alpha,
            p=p,
            target="param",
        )

    return cilo_idset, cihi_idset, cilo_param, cihi_param, mu_bar, sigma_sq_bar


def _draw_data(
    n_obs: int,
    p: float,
    beta_params: tuple[float, float],
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw data."""
    y = rng.beta(beta_params[0], beta_params[1], n_obs)
    w = rng.binomial(1, p, n_obs)

    return y, w


def _compute_ci(
    alpha: float,
    p: float,
    mu_bar: float,
    sigma_sq_bar: float,
    n_obs: int,
    target: str,
) -> tuple[float, float]:
    if target == "idset":
        z = norm.ppf((alpha + 1) / 2)
    elif target == "param":
        z = norm.ppf(alpha)

    lower = (mu_bar - z * (np.sqrt(sigma_sq_bar) / np.sqrt(p * n_obs))) * p
    upper = (mu_bar + z * (np.sqrt(sigma_sq_bar) / np.sqrt(p * n_obs))) * p + 1 - p

    return lower, upper


def _compute_bootstrap_ci(
    n_boot: int,
    y: np.ndarray,
    w: np.ndarray,
    alpha: float,
    p: float,
    target: str,
    rng: np.random.Generator = RNG,
) -> tuple[float, float]:
    boot_mu_bar = np.zeros(n_boot)
    y_nomiss = y[w == 1]

    for b in range(n_boot):
        boot_y_nomiss = rng.choice(y_nomiss, size=len(y_nomiss), replace=True)
        boot_mu_bar[b] = np.mean(boot_y_nomiss)

    if target == "idset":
        mu_lo, mu_hi = np.quantile(boot_mu_bar, [(1 - alpha) / 2, 1 - (1 - alpha) / 2])
    elif target == "param":
        mu_lo, mu_hi = np.quantile(boot_mu_bar, [1 - alpha, alpha])

    lower = mu_lo * p
    upper = mu_hi * p + 1 - p

    return lower, upper
