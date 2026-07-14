import numpy as np
import pytest

from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import SCALAR_PROBLEMS

BACKEND = SciPyBackend()
START = "standard"
SCALAR_CAPABLE_METHODS = ["BFGS", "L-BFGS-B"]
LEAST_SQUARES_METHODS = ["lm", "trf", "dogbox"]


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=lambda p: p.id)
@pytest.mark.parametrize("method", SCALAR_CAPABLE_METHODS)
def test_solves_a_scalar_cost_problem_to_the_known_optimum(problem, method):
    result = BACKEND.solve(problem, method, START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED, result.status
    assert np.allclose(result.x_final, problem.optima[0].x_star, atol=1e-3)
    assert np.isclose(result.cost_final, problem.optima[0].cost_star, atol=1e-3)


@pytest.mark.parametrize("problem", SCALAR_PROBLEMS, ids=lambda p: p.id)
@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_least_squares_methods_reject_a_scalar_cost_problem(problem, method):
    # lm/trf/dogbox fundamentally need a residual vector (they call
    # least_squares, which computes J.T@r internally); a kind="scalar"
    # problem's residual() is already the cost, not a residual vector,
    # so routing it there would silently misinterpret the objective
    # rather than raise - reject it clearly instead.
    with pytest.raises(ValueError, match="scalar"):
        BACKEND.solve(problem, method, START, RunConfig(max_iter=200))
