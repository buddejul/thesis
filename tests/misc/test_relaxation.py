"""Test relaxation functions."""
from thesis.config import RNG
from thesis.misc._task_relaxation_mte import solve_lp_convex


def test_solve_lp_convex_runs() -> None:
    beta = RNG.uniform(-1, 1)

    solve_lp_convex(beta=beta)
