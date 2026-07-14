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


def test_a_looser_tolerance_does_not_stall_before_making_real_progress():
    # tolr bounds how small the relative change between successive
    # accepted iterates can get before the solver treats it as "no
    # progress" and gives up - it is not a precision knob. Mapping
    # RunConfig.tolerance onto it made a *looser* tolerance produce a
    # *worse* result: at tolerance=0.1, the very first rejected step's
    # damping increase already looked like a stall (measured: STALLED
    # after 1 iteration, dist_to_opt~1.95), while tolerance=0.01 (a
    # tighter, not looser, request) converged with dist_to_opt~0.078.
    # The 1.0 bound is a generous margin above the ~0.66 this actually
    # measures under trust's own damping schedule - the point is that it
    # converges at all, not an exact precision claim at this tolerance.
    result = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200, tolerance=0.1))

    assert result.status is RunStatus.CONVERGED
    assert result.dist_to_opt < 1.0
