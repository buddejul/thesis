"""Tests for Imbens Manski simulations."""
from thesis.config import RNG


def test_compute_bootstrap_ci() -> None:
    """Test compute_bootstrap_ci."""
    RNG.normal(0, 1, 100)
