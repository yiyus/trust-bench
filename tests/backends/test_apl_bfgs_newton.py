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


def test_capabilities_declares_bfgs_and_trust_exact():
    methods = BACKEND.capabilities().methods

    assert "BFGS" in methods
    assert "trust-exact" in methods
    assert methods["trust-exact"].analytic_hessian is True


def test_solve_with_bfgs_method_converges():
    result = BACKEND.solve(PROBLEM, "BFGS", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED
    assert result.x_final == pytest.approx([1.0, 1.0], abs=1e-4)
    assert result.n_heval == 0


def test_solve_with_trust_exact_method_converges_and_reports_n_heval():
    result = BACKEND.solve(PROBLEM, "trust-exact", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED
    assert result.x_final == pytest.approx([1.0, 1.0], abs=1e-4)
    assert result.n_heval is not None
    assert result.n_heval > 0


def test_trust_exact_converges_in_fewer_iterations_than_bfgs():
    bfgs = BACKEND.solve(PROBLEM, "BFGS", START, RunConfig(max_iter=200))
    newton = BACKEND.solve(PROBLEM, "trust-exact", START, RunConfig(max_iter=200))

    assert bfgs.status is RunStatus.CONVERGED
    assert newton.status is RunStatus.CONVERGED
    assert newton.n_iter < bfgs.n_iter
