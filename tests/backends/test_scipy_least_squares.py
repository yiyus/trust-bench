import numpy as np
import pytest

from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunStatus
from trust_bench.problems import quadratic

BACKEND = SciPyBackend()
PROBLEM = quadratic.PROBLEM
START = "standard"
LEAST_SQUARES_METHODS = ["lm", "trf", "dogbox"]


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_method_solves_the_trivial_quadratic_to_the_known_optimum(method):
    result = BACKEND.solve(PROBLEM, method, START, {"max_iter": 100})

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_capabilities_bounds_flag_matches_which_methods_accept_bounds():
    methods = BACKEND.capabilities().methods

    assert methods["lm"].bounds is False
    assert methods["trf"].bounds is True
    assert methods["dogbox"].bounds is True


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_bounded_methods_accept_and_respect_box_constraints(method):
    result = BACKEND.solve(
        PROBLEM, method, START, {"max_iter": 100, "bounds": ([0.5, -np.inf], [np.inf, np.inf])}
    )

    assert result.x_final[0] >= 0.5 - 1e-9
    assert np.isclose(result.x_final[0], 0.5, atol=1e-6)


def test_lm_rejects_box_constraints():
    with pytest.raises(ValueError):
        BACKEND.solve(
            PROBLEM, "lm", START, {"max_iter": 100, "bounds": ([0.5, -np.inf], [np.inf, np.inf])}
        )
