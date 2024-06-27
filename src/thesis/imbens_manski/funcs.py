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
    rng: np.random.Generator = RNG,
    beta_params: tuple[float, float] = (1, 1),
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
            _experiment(n_obs, p, alpha, beta_params=beta_params, rng=rng)
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
