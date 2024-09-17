"""Utilities used in various parts of the project."""

import inspect
from collections.abc import Callable
from pathlib import Path

import yaml  # type: ignore[import-untyped]


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
