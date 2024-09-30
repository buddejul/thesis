"""All the general configuration of the project."""
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

GROUPS = ["marital_status", "qualification"]

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]

RNG = np.random.default_rng()
