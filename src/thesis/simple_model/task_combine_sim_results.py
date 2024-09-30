"""Task for combining simulation results into dataset."""

from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]  # type: ignore[import-untyped]
from pytask import Product

from thesis.config import BLD, PATH_TO_SIM_RESULTS
from thesis.simple_model.task_simple_model_sims import (
    EPS_FUNS_NUMERICAL_DELTA,
)
from thesis.utilities import get_func_as_string

EPS_FUNS_STRINGS = [get_func_as_string(eps_fun) for eps_fun in EPS_FUNS_NUMERICAL_DELTA]
KAPPA_FUNS_STRINGS = [
    get_func_as_string(kappa_fun) for kappa_fun in EPS_FUNS_NUMERICAL_DELTA
]

# Find all files in PATH_TO_SIM_RESULTS including all subdirectories.
files = list(Path(PATH_TO_SIM_RESULTS).rglob("*.pkl"))


def task_combine_sim_results(
    sim_results: list[Path] = files,
    path_to_data: Annotated[Path, Product] = Path(
        BLD / "simple_model" / "sim_results_combined.pkl",
    ),
) -> None:
    """Combine all simulation results into single dataframe."""
    # Get all files in path_to_sim_results.
    # Each file will consist of a dictionary with the following keys:
    # - "settings": a dictionary with settings used for the simulation
    # - "data": a pandas DataFrame with the simulation results

    # The settings will be used to create the columns of the final DataFrame.
    # The data will be used to create the rows of the final DataFrame.
    # The final DataFrame will be saved as a pickle file at path_to_data.

    sr_to_combine = []

    for file in sim_results:
        _res = pd.read_pickle(file)  # noqa: S301
        _settings = _res["settings"]
        _data = _res["data"]

        _data["coverage"] = (_data["lo"] <= _data["true"]) & (
            _data["hi"] >= _data["true"]
        )

        _data["length"] = _data["hi"] - _data["lo"]

        _data["hi_covers"] = _data["hi"] >= _data["true"]
        _data["lo_covers"] = _data["lo"] <= _data["true"]

        sr = _data.mean()

        _keys_to_skip = ["rng", "bootstrap_params", "local_ates"]
        for key, value in _settings.items():
            if key in _keys_to_skip:
                continue
            sr._set_value(key, value)

        _eps_fun_str = _clean_lambda_string(_settings["bootstrap_params"]["eps_fun"])
        _kappa_fun_str = _clean_lambda_string(
            _settings["bootstrap_params"]["kappa_fun"],
        )

        sr._set_value("eps_fun", _eps_fun_str)
        sr._set_value("kappa_fun", _kappa_fun_str)
        sr._set_value("late_complier", _settings["local_ates"].complier)

        sr_to_combine.append(sr)

    data = pd.DataFrame(sr_to_combine)

    for col in ["n_obs", "n_boot", "n_sims"]:
        data[col] = data[col].astype(int)

    data.to_pickle(path_to_data)


def _clean_lambda_string(funstr: str) -> str:
    funstr = funstr.split("lambda n: ")[1]
    funstr = funstr.split("\\n")[0]
    funstr = funstr.replace(" ", "")
    funstr = funstr.replace("**", "pow")
    funstr = funstr.replace("*", "x")
    funstr = funstr.replace("/", "div")
    return funstr.replace(",", "")
