"""Functions for analysis of Bhattacharya 2009."""

from collections.abc import Callable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from scipy.stats import gaussian_kde, norm  # type: ignore[import-untyped]

from thesis.bhattacharya.cdd_funcs import find_extreme_points_box


def draw_data(
    num_obs: int,
    c_1: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw num_obs observations from N(c_1, sigma^2)."""
    return rng.normal(c_1, sigma, num_obs)


def v_hat(data: np.ndarray) -> float:
    """Calculate the Bhattacharya 2009 estimator for a given sample."""
    mean = np.mean(data)
    return mean * (mean < 0)


def sim_distribution_v_hat(
    num_reps: int,
    num_obs: int,
    c_1: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate finite sample distribution of v_hat."""
    out = np.empty(num_reps)

    for i in range(num_reps):
        data = draw_data(num_obs, c_1, sigma, rng)
        out[i] = v_hat(data)

    return out


def plot_scaled_distr(
    num_obs: int,
    num_reps: int,
    num_grid: int,
    c_1: float,
    sigma: float,
    rng: np.random.Generator,
) -> go.Figure:
    """Plot the scaled distribution of v_hat."""
    grid = np.linspace(-3, 3, num_grid)

    distr = sim_distribution_v_hat(
        num_reps=num_reps,
        num_obs=num_obs,
        c_1=c_1,
        sigma=sigma,
        rng=rng,
    )

    v = np.min(c_1, 0)

    scaled_distr = np.sqrt(num_obs) * (distr - v)

    try:
        kde = gaussian_kde(scaled_distr).evaluate(grid)
    except np.linalg.LinAlgError:
        kde = np.zeros(num_grid)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=kde,
            mode="lines",
            name="Finite Sample (KDE)",
            line={"dash": "solid", "color": "blue"},
        ),
    )

    # Add N(0,sigma^2) distribution

    sigma_asympt = sigma if c_1 <= 0 else 0

    if c_1 < 0:
        asym_distr = norm.pdf(grid, loc=0, scale=sigma_asympt)
    if c_1 == 0:
        # Minimum of norm.pdf(grid, loc=0, scale=sigma_asympt) and 0
        asym_distr = np.where(grid <= 0, norm.pdf(grid, loc=0, scale=sigma_asympt), 0)
        # Add zero to grid and set asym_distr to 0.5 at that point
        grid = np.concatenate((np.zeros(1), grid))
        asym_distr = np.concatenate(([0.5], asym_distr))

        idx = np.argsort(grid)
        grid = grid[idx]
        asym_distr = asym_distr[idx]
    if c_1 > 0:
        asym_distr = np.zeros(num_grid)

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=asym_distr,
            mode="lines",
            name="Asymptotic Distribution",
            line={"dash": "solid", "color": "red"},
        ),
    )

    # Add mean(scaled_distr) and alpha/2 and 1-alpha/2 quantiles
    alpha = 0.05

    mean = np.mean(scaled_distr)
    q1_fs = np.quantile(scaled_distr, alpha / 2)
    q2_fs = np.quantile(scaled_distr, 1 - alpha / 2)

    fig.add_trace(
        go.Scatter(
            x=[mean, mean],
            y=[0, 0.5],
            mode="lines",
            name=f"FS: Mean = {mean:.2f}",
            line={"dash": "dash", "color": "orange"},
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[q1_fs, q1_fs],
            y=[0, 0.5],
            mode="lines",
            name=f"FS: {alpha/2} quantile = {q1_fs:.2f}",
            line={"dash": "dash", "color": "green"},
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[q2_fs, q2_fs],
            y=[0, 0.5],
            mode="lines",
            name=f"FS: {1-alpha/2} quantile = {q2_fs:.2f}",
            line={"dash": "dash", "color": "green"},
        ),
    )

    # Add quantiles of the asymptotic distribution
    q1_asym = norm.ppf(alpha / 2, loc=0, scale=sigma_asympt) if c_1 <= 0 else 0

    q2_asym = norm.ppf(1 - alpha / 2, loc=0, scale=sigma_asympt) if c_1 < 0 else 0

    fig.add_trace(
        go.Scatter(
            x=[q1_asym, q1_asym],
            y=[0, 0.5],
            mode="lines",
            name=f"Asym.: {alpha/2} quantile = {q1_asym:.2f}",
            line={"dash": "dot", "color": "black"},
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=[q2_asym, q2_asym],
            y=[0, 0.5],
            mode="lines",
            name=f"Asym.: {1-alpha/2} quantile = {q2_asym:.2f}",
            line={"dash": "dot", "color": "black"},
        ),
    )

    fig.update_layout(
        title=f"N = {num_obs}, c_1 = {c_1}, Sigma = {sigma}, Simulations = {num_reps}",
        xaxis_title="sqrt(n) * (v_hat - v)",
        yaxis_title="Density",
    )

    return fig


def bhatta_confidence_interval(
    data: np.ndarray,
    c_n: Callable | float,
    n_reps: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calculate the Bhattacharya 2009 confidence interval for a given sample."""
    num_obs = len(data)

    # Step 1: Construct set of optimal solutions up to some tolerance
    gamma_1_hat = np.mean(data)
    gamma_2_hat = 0.5
    gamma = np.array([gamma_1_hat, gamma_2_hat])

    basic_feasible_solutions = find_extreme_points_box(2)[:, 1:]

    values = basic_feasible_solutions @ gamma.T

    v_hat = np.min(values)

    c_n_val = c_n(num_obs) if callable(c_n) else c_n

    idx = np.where(v_hat + c_n_val >= values)[0]

    num_solutions = len(idx)

    estimated_optimal_solutions = basic_feasible_solutions[idx, :]

    # Step 2: For each of B draws w from N(0, sigma^2_hat) calculate the minimum
    # value over the estimated optimal solutions
    sigma_hat = np.std(data)

    asym_distr = np.empty(n_reps)

    for i in range(n_reps):
        w1 = rng.normal(0, sigma_hat)
        w2 = 0
        w = np.array([w1, w2])

        if len(idx) == 1:
            asym_distr[i] = np.dot(estimated_optimal_solutions, w)[0]
        else:
            _values = estimated_optimal_solutions @ w

            asym_distr[i] = np.min(_values)

    z_hi = np.quantile(asym_distr, 1 - alpha / 2)
    z_lo = np.quantile(asym_distr, alpha / 2)

    ci_lo = v_hat - z_hi / np.sqrt(num_obs)
    ci_hi = v_hat - z_lo / np.sqrt(num_obs)

    return np.array([ci_lo, ci_hi, z_lo, z_hi, v_hat, num_solutions])


# Run a small simulation to see how to confidence intervals look on average
# Choosing c_n large amounts to correctly choosing both solutions when c_1 = 0
# Choosing c_n = 0 amounts to choosing (0, 0) whenever c_1_hat > 0 and (1, 0)
# whenever c_1_hat < 0


def sim_confidence_interval(
    num_obs: int,
    c_1: float,
    c_n: float,
    sigma: float,
    num_sims: int,
    alpha: float,
    ci_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate finite sample distribution of v_hat."""
    allowed_ci_type = ["b2009"]

    if ci_type not in allowed_ci_type:
        msg = f"ci_type must be one of {allowed_ci_type} but got {ci_type}."
        raise ValueError(msg)

    res = np.zeros((num_sims, 6))

    for i in range(num_sims):
        data = draw_data(num_obs, c_1, sigma, rng)

        if ci_type == "b2009":
            res[i, :] = bhatta_confidence_interval(
                data=data,
                c_n=c_n,
                n_reps=1000,
                alpha=alpha,
                rng=rng,
            )

    cols = ["ci_lo", "ci_hi", "z_lo", "z_hi", "v_hat", "num_solutions"]

    return pd.DataFrame(res, columns=cols)


def c_n_normal(num_obs: int, sigma: float, alpha: float) -> float:
    """Tuning parameter c_n corresponding to N(0, sigma^2) level alpha test."""
    return sigma * norm.ppf(1 - alpha / 2) / (np.sqrt(num_obs))
