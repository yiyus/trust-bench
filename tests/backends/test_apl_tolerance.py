import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import rosenbrock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
PROBLEM = rosenbrock.PROBLEM
START = "standard"


def test_solve_converges_more_precisely_with_a_tighter_tolerance():
    loose = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200, tolerance=1e-2))
    tight = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200, tolerance=1e-10))

    assert loose.status is RunStatus.CONVERGED
    assert tight.status is RunStatus.CONVERGED
    assert tight.dist_to_opt < loose.dist_to_opt
