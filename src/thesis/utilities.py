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


def bic(n: int) -> float:
    """Tuning parameter sequence based on BIC."""
    return np.sqrt(np.log(n))
