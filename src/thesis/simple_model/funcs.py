"""Functions for bootstrap sampling experiment."""

from collections.abc import Callable
from functools import partial
from math import comb

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from statsmodels.api import add_constant  # type: ignore[import-untyped]
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore[import-untyped]

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
    bootstrap_params: dict[str, Callable] | None = None,
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
        bootstrap_params: Additional parameters for the bootstrap method depending on
            the bootstrap_method.

    Returns:
        DataFrame with the results of the simulation.

    """
    if bootstrap_params is None:
        bootstrap_params = {}
    _check_constraint_supported(constraint_mtr)
    _check_bootsrap_method_supported(bootstrap_method)
    _check_instrument_and_u_hi_consistent(u_hi, instrument)
    _check_complier_always_taker_consistent(local_ates, constraint_mtr)
    _check_bootstrap_params_supplied(bootstrap_method, bootstrap_params)

    results = np.zeros((n_sims, 2))

    for i in range(n_sims):
        data = _draw_data(n_obs, local_ates=local_ates, instrument=instrument, rng=rng)

        results[i] = _bootstrap_ci(
            n_boot=n_boot,
            bootstrap_method=bootstrap_method,
            bootstrap_params=bootstrap_params,
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
    bootstrap_params: dict[str, Callable] | None = None,
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

    if bootstrap_params is None:
        bootstrap_params = {}
    if bootstrap_method == "standard":
        return _ci_standard_bootstrap(
            n_boot=n_boot,
            data=data,
            alpha=alpha,
            u_hi=u_hi,
            constraint_mtr=constraint_mtr,
            rng=rng,
        )
    if bootstrap_method == "numerical_delta":
        return _ci_numerical_delta_bootstrap(
            n_boot=n_boot,
            data=data,
            alpha=alpha,
            u_hi=u_hi,
            constraint_mtr=constraint_mtr,
            rng=rng,
            eps_fun=bootstrap_params["eps_fun"],
        )
    if bootstrap_method == "analytical_delta":
        return _ci_analytical_delta_bootstrap(
            n_boot=n_boot,
            data=data,
            alpha=alpha,
            u_hi=u_hi,
            constraint_mtr=constraint_mtr,
            rng=rng,
            kappa_fun=bootstrap_params["kappa_fun"],
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
        boot_data, pscores_boot = _draw_bootstrap_data(data=data, n_obs=n_obs, rng=rng)

        # TODO(@buddejul): This might be more efficiently implemented by vectorizing the
        # _idset call on the _late and _estimate_pscores bootstrap estimates.
        boot_lo[i], boot_hi[i] = _idset(
            b_late=_late(boot_data),
            u_hi=u_hi,
            pscores_hat=pscores_boot,
            constraint_mtr=constraint_mtr,
        )

    # Explicitly make this a float to avoid static typing issues. np.quantile returns a
    # np.float64 type, which is not compatible with the float type hint, although this
    # seems to be addressed in https://github.com/numpy/numpy/pull/27334.
    # Note we also take (1 - ) alpha/2 quantiles. This is to keep to code consistent.
    # Following the usual Imbens and Manski argument, when covering the parameter we
    # would only use (1 - ) alpha quantiles.
    return float(np.quantile(boot_lo, alpha / 2)), float(
        np.quantile(boot_hi, 1 - alpha / 2),
    )


def _ci_numerical_delta_bootstrap(
    n_boot: int,
    data: np.ndarray,
    alpha: float,
    u_hi: float,
    constraint_mtr: str,
    rng: np.random.Generator,
    eps_fun: Callable = lambda n: n ** (-1 / 6),
) -> tuple[float, float]:
    """Compute the numerical delta bootstrap confidence interval.

    Based on Hong and Li (2018), for details see p. 382.

    The stepsize is set to n^(-1/6) by default, which is at least theoretically
    consistent with eps -> zero and rn * eps -> infty, i.e. eps converges more slowly.

    """
    n_obs = data.shape[0]

    eps = eps_fun(n_obs)

    rn = np.sqrt(n_obs)

    late = _late(data)
    _id_data = _idset(late, u_hi, _estimate_pscores(data), constraint_mtr)

    boot_delta_lo = np.zeros(n_boot)
    boot_delta_hi = np.zeros(n_boot)

    for i in range(n_boot):
        # Step 1: Draw Z_s from the bootstrap distribution.
        boot_data, pscores_boot = _draw_bootstrap_data(data=data, n_obs=n_obs, rng=rng)

        z_s = rn * (_late(boot_data) - late)

        # Step 2: Compute the numerical derivative.
        # Note that in our case we need to do this for both the lower and upper bound.

        _id_plus_eps_boot = _idset(
            b_late=late + eps * z_s,
            u_hi=u_hi,
            pscores_hat=pscores_boot,
            constraint_mtr=constraint_mtr,
        )

        boot_delta_lo[i], boot_delta_hi[i] = (1 / eps) * (_id_plus_eps_boot - _id_data)

    # Construct 1 - alpha two-sided equal-tailed confidence interval for phi(beta_s).
    # Note: Based on the Imbens and Manski argument for covering the parameter - instead
    # of the identified set - we might only need to take "alpha" critical values here,
    # instead of alpha. To achieve this, the user may supply alpha * 2 as the alpha.

    # Compute the lower CI bound for the lower bound of the identified set.
    ci_lo = _id_data[0] - (1 / rn) * np.quantile(boot_delta_lo, 1 - alpha / 2)

    # Compute the upper CI bound for the upper bound of the identified set.
    ci_hi = _id_data[1] - (1 / rn) * np.quantile(boot_delta_hi, alpha / 2)

    return ci_lo, ci_hi


def _ci_analytical_delta_bootstrap(
    n_boot: int,
    data: np.ndarray,
    alpha: float,
    u_hi: float,
    constraint_mtr: str,
    rng: np.random.Generator,
    kappa_fun: Callable = lambda n: n ** (1 / 6),
) -> tuple[float, float]:
    """Compute the analytical delta bootstrap confidence interval.

    Based on Fang and Santos (2017), adapted from Example 2.1 equation (26). kappa_fun
    is the tuning parameter used for the pretest to estimate the derivative.

    """
    n_obs = data.shape[0]

    kappa_n = kappa_fun(n_obs)

    rn = np.sqrt(n_obs)

    # Estimate late using 2SLS to get the standard error for estimating the derivative.
    late, se_late = _late_2sls(data)

    pscores = _estimate_pscores(data)

    boot_late_scaled_and_centered = np.zeros(n_boot)

    w = (pscores[1] - pscores[0]) / (u_hi + pscores[1] - pscores[0])

    # Step 1: Bootstrap quantiles for the identified parameter beta_s.
    for i in range(n_boot):
        # Step 1: Draw Z_s from the bootstrap distribution of beta_s.
        boot_data, _ = _draw_bootstrap_data(data=data, n_obs=n_obs, rng=rng)

        boot_late_scaled_and_centered[i] = rn * (_late(boot_data) - late)

    # Step 2: Estimate the derivative.
    # We need two separate derivatives, since the upper and lower bound have different
    # solutions.
    _di_phi = partial(
        _d_phi_kink,
        beta_late=late,
        se_late=se_late,
        kappa_n=kappa_n,
        rn=rn,
    )
    if constraint_mtr == "none":
        d_phi_upper = partial(_di_phi, slope_left=1, slope_right=1)
        d_phi_lower = d_phi_upper

    elif constraint_mtr == "increasing":
        d_phi_upper = partial(_di_phi, slope_left=1, slope_right=w)
        d_phi_lower = partial(_di_phi, slope_left=w, slope_right=1)

    # Step 3: Apply derivative to bootstrap quantiles to get the confidence interval.
    # In our special case we know the function is monotonically increasing, hence we can
    # compute the bootstrap percentile first and then apply the estimated derivative.
    id_lo, id_hi = _idset(
        b_late=late,
        u_hi=u_hi,
        pscores_hat=pscores,
        constraint_mtr=constraint_mtr,
    )

    _c_alpha_half = d_phi_lower(np.quantile(boot_late_scaled_and_centered, alpha / 2))
    boot_ci_lo = id_lo - _c_alpha_half / rn

    _c_1_minus_alpha_half = d_phi_upper(
        np.quantile(boot_late_scaled_and_centered, 1 - alpha / 2),
    )
    boot_ci_hi = id_hi + _c_1_minus_alpha_half / rn

    return boot_ci_lo, boot_ci_hi


def _idset(
    b_late: float,
    u_hi: float,
    pscores_hat: tuple[float, float],
    constraint_mtr: str,
) -> np.ndarray:
    w = (pscores_hat[1] - pscores_hat[0]) / (u_hi + pscores_hat[1] - pscores_hat[0])

    if constraint_mtr == "none":
        lo = w * b_late - (1 - w)
        hi = w * b_late + (1 - w)

    # Restricting the MTRs to be increasing changes the solution to have a kink when
    # viewed as a function of the identified parameter.
    elif constraint_mtr == "increasing":
        hi = _phi_kink(theta=b_late, kink=0, slope_left=1, slope_right=w) + (1 - w)
        lo = _phi_kink(theta=b_late, kink=0, slope_left=w, slope_right=1) - (1 - w)

    return np.array([lo, hi])


def _late(data: np.ndarray) -> float:
    """Estimate the LATE using the Wald estimator."""
    yz1 = data[data[:, 2] == 1, 0].mean()
    yz0 = data[data[:, 2] == 0, 0].mean()

    dz1 = data[data[:, 2] == 1, 1].mean()
    dz0 = data[data[:, 2] == 0, 1].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def _late_2sls(data: np.ndarray) -> tuple[float, float]:
    """Estimate the LATE using 2SLS; also returns standard error."""
    # Data order is y, d, z, u.

    # In Statsmodels endog refers to the dependent variable, while exog refers to the
    # explanatory variables, including all controls and the - endogenous - treatment
    # variable.
    model = IV2SLS(
        endog=data[:, 0],
        exog=add_constant(data[:, 1]),
        instrument=add_constant(data[:, 2]),
    )

    res = model.fit()

    return res.params[1], res.bse[1]


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


def _draw_bootstrap_data(
    data: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Draw data for the bootstrap; redraw if propensity scores are equal."""
    boot_data = data[rng.choice(n_obs, size=n_obs, replace=True)]

    pscores_boot = _estimate_pscores(boot_data)

    if pscores_boot[0] != pscores_boot[1]:
        return boot_data, pscores_boot

    # If the pscores are not the same we redraw the data. We set a maximum number of
    # iterations to avoid infinite loops.
    i = 0
    while pscores_boot[0] == pscores_boot[1]:
        boot_data = data[rng.choice(n_obs, size=n_obs, replace=True)]
        pscores_boot = _estimate_pscores(boot_data)
        i = i + 1

        if i == max_iter:
            msg = (
                f"Bootstrap failed to generate different propensity scores"
                f"({max_iter} draws)."
            )
            raise ValueError(msg)

    return boot_data, pscores_boot


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
    slope_left: float,
    slope_right: float,
) -> float:
    return slope_left * theta * (theta < kink) + slope_right * theta * (theta >= kink)


def _d_phi_kink(
    h: float,
    beta_late: float,
    se_late: float,
    kappa_n: float,
    rn: float,
    slope_left: float,
    slope_right: float,
    kink: float = 0,
) -> float:
    """Estimator for the derivative of the identified set."""
    # TODO(@buddejul): Wrifte more general version allowing for different kink point.
    # Currently we do this for a kink at zero; this should show up in the pre-test.

    cond_right = rn * (beta_late - kink) / se_late > kappa_n
    cond_left = rn * (beta_late - kink) / se_late < -kappa_n
    cond_mid = ~cond_right & ~cond_left

    return (
        cond_right * h * slope_right
        + cond_left * h * slope_left
        + cond_mid * np.max([h * slope_right, -h * slope_left])
    )


def _check_constraint_supported(constraint_mtr: str) -> None:
    supported = ["none", "increasing"]
    if constraint_mtr not in supported:
        msg = (
            f"Constraint '{constraint_mtr}' not supported.\n"
            f"Supported constraints: {supported}"
        )
        raise ValueError(msg)


def _check_bootsrap_method_supported(bootstrap_method: str) -> None:
    supported = ["standard", "numerical_delta", "analytical_delta"]
    if bootstrap_method not in supported:
        msg = (
            f"Bootstrap method '{bootstrap_method}' not supported.\n"
            f"Supported constraints: {supported}"
        )
        raise ValueError(msg)


def _check_instrument_and_u_hi_consistent(u_hi: float, instrument: Instrument) -> None:
    if u_hi + instrument.pscores[1] > 1:
        msg = (
            f"Upper bound u_hi + pscores[1] = {u_hi + instrument.pscores[1]} "
            f"exceeds 1. This is not allowed."
        )
        raise ValueError(msg)


def _check_complier_always_taker_consistent(local_ates, constraint_mtr):
    if constraint_mtr == "increasing" and (
        local_ates.complier < 0 and local_ates.always_taker >= 1
    ):
        # To be consistent with increasing MTR assumptions, the always-taker can be at
        # most min(1, 1 + complier ate). Hence, if the complier ATE is negative, the
        # always-taker ATE needs to be smaller than 1.
        msg = (
            "Whenever late_complier < 0, the largest possible always-taker ATE is "
            "1 + later_complier < 1."
        )
        raise ValueError(msg)


def _check_bootstrap_params_supplied(
    bootstrap_method: str,
    bootstrap_params: dict[str, Callable],
) -> None:
    if bootstrap_method == "numerical_delta" and "eps_fun" not in bootstrap_params:
        msg = (
            "Numerical delta bootstrap method requires the user to supply an epsilon "
            "function via bootstrap_params under key 'eps_fun'."
        )
        raise ValueError(msg)

    if bootstrap_method == "analytical_delta" and "kappa_fun" not in bootstrap_params:
        msg = (
            "Analytical delta bootstrap method requires the user to supply a kappa "
            "function via bootstrap_params under key 'kappa_fun'."
        )
        raise ValueError(msg)
