"""Utilities used in various parts of the project."""

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]

from thesis.classes import Instrument, LocalATEs


def read_yaml(path):
    """Read a YAML file.

    Args:
        path (str or pathlib.Path): Path to file.

    Returns:
        dict: The parsed YAML file.

    """
    with Path.open(path) as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            info = (
                "The YAML file could not be loaded. Please check that the path points "
                "to a valid YAML file."
            )
            raise ValueError(info) from error
    return out


def get_func_as_string(func: Callable) -> str:
    """Inspects a function and converts the first body line into a string."""
    func_str = str(inspect.getsourcelines(func)[0])
    func_str = func_str.split(":")[1].strip()
    func_str = func_str.removesuffix(")\\n']")
    func_str = func_str.removesuffix(",\\n']")
    func_str = func_str.replace(" ", "")
    func_str = func_str.replace("**", "pow")
    func_str = func_str.replace("*", "x")
    return func_str.replace("/", "div")


def draw_data(
    n_obs: int,
    local_ates: LocalATEs,
    instrument: Instrument,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw data for given local ates and instrument."""
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


def make_mtr_binary_iv(
    yd_c: float | np.ndarray,
    yd_at: float | np.ndarray,
    yd_nt: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
):
    """Construct MTR with constant splines for binary IV."""
    _pscores = {
        "pscore_lo": pscore_lo,
        "pscore_hi": pscore_hi,
    }

    def mtr(u):
        return (
            yd_at * _at(u, **_pscores)
            + yd_c * _c(u, **_pscores)
            + yd_nt * _nt(u, **_pscores)
        )

    return mtr


def _at(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    del pscore_hi
    return u <= pscore_lo


def _c(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    return pscore_lo <= u and pscore_hi > u


def _nt(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    del pscore_lo
    return u >= pscore_hi


def simulate_data_from_mtrs_binary_iv(
    mtr0: Callable,
    mtr1: Callable,
    num_obs: int,
    rng: np.random.Generator,
    iv_support: np.ndarray,
    iv_pmf: np.ndarray,
    iv_pscores: np.ndarray,
    sigma: float = 0.1,
) -> dict[str, np.ndarray]:
    """Simulate data from MTRs for the binary-IV model."""
    # TODO: This is hard-coded, provide as argument.
    choices = np.hstack([iv_support.reshape(-1, 1), iv_pscores.reshape(-1, 1)])

    idx = rng.choice(iv_support, size=num_obs, p=iv_pmf)

    data = choices[idx]

    z = np.array(data[:, 0], dtype=int)
    pscores = data[:, 1]

    u = rng.uniform(size=num_obs)
    d = u < pscores

    y = np.empty(num_obs)
    idx_d0 = d == 0

    y[idx_d0] = mtr0(u[idx_d0]) + rng.normal(scale=sigma, size=np.sum(idx_d0))

    y[~idx_d0] = mtr1(u[~idx_d0]) + rng.normal(scale=sigma, size=np.sum(~idx_d0))

    return {"z": z, "d": d, "y": y, "u": u}


def constraint_dict_to_string(d: dict) -> str:
    """Convert dictionary with shape constraints to string for tasks."""
    parts = []
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            value_str = ", ".join(value)
        elif value is None:
            value_str = "None"
        else:
            value_str = value
        parts.append(f"{key}: {value_str}")

    if not parts:
        return "none"

    return "; ".join(parts)


def bic(n: int) -> float:
    """Tuning parameter sequence based on BIC."""
    return np.sqrt(np.log(n))
