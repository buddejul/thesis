"""Test functions for bootstrap_fail."""

import pandas as pd  # type: ignore[import-untyped]
from thesis.bootstrap_fail.funcs import _draw_data
from thesis.config import RNG


def test_data_moments_boundary() -> None:
    n_obs = 100_000

    expected = pd.DataFrame(
        {
            "y_given_z": [0.0, 0.0],
            "d_given_z": [0.4, 0.6],
        },
        index=[0, 1],
    )

    data = pd.DataFrame(
        _draw_data(n_obs, RNG, boundary=True),
        columns=["y", "d", "z", "u"],
    )

    actual = pd.DataFrame(
        {
            "y_given_z": data.groupby("z")["y"].mean(),
            "d_given_z": data.groupby("z")["d"].mean(),
        },
        index=[0, 1],
    )

    pd.testing.assert_frame_equal(actual, expected, atol=0.01)
