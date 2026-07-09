import dataclasses

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


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_finite_difference_derivative_mode_never_calls_the_analytic_jacobian(method):
    calls = []

    def counting_jacobian(x):
        calls.append(x)
        return PROBLEM.jacobian(x)

    problem = dataclasses.replace(PROBLEM, jacobian=counting_jacobian)

    result = BACKEND.solve(
        problem, method, START, {"max_iter": 100, "derivative_mode": "finite-difference"}
    )

    assert calls == []
    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_solve_falls_back_to_finite_difference_when_problem_has_no_analytic_jacobian():
    problem = dataclasses.replace(PROBLEM, jacobian=None)

    result = BACKEND.solve(problem, "lm", START, {"max_iter": 100})

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_capabilities_losses_match_which_methods_accept_non_linear_losses():
    methods = BACKEND.capabilities().methods

    assert methods["lm"].losses == frozenset({"linear"})
    assert "huber" in methods["trf"].losses
    assert "huber" in methods["dogbox"].losses


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_loss_config_reaches_scipy_for_bounded_methods(method):
    # An unrecognised loss name is rejected by scipy itself; this only
    # happens if solve() actually forwards config["loss"] to
    # least_squares(loss=...), so a solve() that silently ignores the key
    # would not raise here.
    with pytest.raises(ValueError):
        BACKEND.solve(PROBLEM, method, START, {"max_iter": 100, "loss": "not-a-real-loss"})


def test_lm_rejects_a_non_linear_loss():
    with pytest.raises(ValueError):
        BACKEND.solve(PROBLEM, "lm", START, {"max_iter": 100, "loss": "huber"})
