import shutil

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import quadratic

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
PROBLEM = quadratic.PROBLEM
START = "standard"

# quadratic.PROBLEM's unconstrained optimum is (0, 0); a lower bound of
# 0.5 on x1 makes that infeasible, so the constrained optimum sits
# exactly on the boundary. Mirrors bounded.py's own _ACTIVE_BOUNDS.
_ACTIVE_BOUNDS = ([0.5, -10.0], [10.0, 10.0])


def test_capabilities_declares_bounds_and_finite_difference_support():
    caps = BACKEND.capabilities().methods["lm"]

    assert caps.bounds is True
    assert "finite-difference" in caps.derivative_modes
    assert "analytic" in caps.derivative_modes


def test_solve_respects_an_active_bound_on_the_trivial_quadratic():
    result = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200, bounds=_ACTIVE_BOUNDS))

    assert result.status is RunStatus.CONVERGED
    assert result.x_final[0] == pytest.approx(0.5, abs=1e-4)
    assert result.x_final[1] == pytest.approx(0.0, abs=1e-4)


def test_solve_with_finite_difference_derivative_mode_uses_more_evaluations_than_analytic():
    analytic = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200))
    fd = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=200, derivative_mode="finite-difference"))

    assert analytic.status is RunStatus.CONVERGED
    assert fd.status is RunStatus.CONVERGED
    assert fd.x_final == pytest.approx(analytic.x_final, abs=1e-4)
    assert fd.n_feval > analytic.n_feval
