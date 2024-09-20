"""Custom classes for thesis."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class Estimand:
    """Target estimand."""

    esttype: str
    u_lo: float | None = None
    u_hi: float | None = None
    dz_cross: tuple[int, int] | None = None


@dataclass
class Instrument:
    """Discrete instrument."""

    support: np.ndarray
    pmf: np.ndarray
    pscores: np.ndarray


@dataclass
class DGP:
    """Data Generating Process."""

    m0: Callable
    m1: Callable
    support_z: np.ndarray
    pmf_z: np.ndarray
    pscores: np.ndarray
    joint_pmf_dz: dict[int, dict[int, float]]

    @property
    def expectation_z(self):
        """Expectation of instrument Z."""
        return np.sum(self.support_z * self.pmf_z)

    @property
    def expectation_d(self):
        """Expectation of binary treatment D."""
        return np.sum(self.pscores * self.pmf_z)

    @property
    def variance_d(self):
        """Variance of binary treatment D."""
        return self.expectation_d * (1 - self.expectation_d)

    @property
    def covariance_dz(self):
        """Covariance of binary treatment D and instrument Z."""
        return np.sum(
            [
                self.joint_pmf_dz[d][z]
                * (d - self.expectation_d)
                * (z - self.expectation_z)
                for d in [0, 1]
                for z in self.support_z
            ],
        )


class MonteCarloSetup(NamedTuple):
    """Setup for Monte Carlo simulations."""

    sample_size: int
    repetitions: int
    u_hi_range: np.ndarray | None = None
