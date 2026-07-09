import dataclasses

import numpy as np
import pytest

from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.result import RunStatus
from trust_bench.problems import quadratic

BACKEND = SciPyBackend()
PROBLEM = quadratic.PROBLEM
START = "standard"
MINIMIZE_METHODS = ["BFGS", "L-BFGS-B", "Newton-CG", "trust-exact", "trust-constr"]
BOUNDED_METHODS = ["L-BFGS-B", "trust-constr"]
UNBOUNDED_METHODS = ["BFGS", "Newton-CG", "trust-exact"]
# Verified directly against scipy: these three accept jac="2-point" (or no
# analytic Jacobian at all); Newton-CG and trust-exact both raise
# "Jacobian is required" when asked to use finite differences.
FD_CAPABLE_METHODS = ["BFGS", "L-BFGS-B", "trust-constr"]
FD_INCAPABLE_METHODS = ["Newton-CG", "trust-exact"]


@pytest.mark.parametrize("method", MINIMIZE_METHODS)
def test_method_solves_the_trivial_quadratic_to_the_known_optimum(method):
    result = BACKEND.solve(PROBLEM, method, START, {"max_iter": 100})

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_capabilities_bounds_flag_matches_which_methods_accept_bounds():
    methods = BACKEND.capabilities().methods

    for method in BOUNDED_METHODS:
        assert methods[method].bounds is True
    for method in UNBOUNDED_METHODS:
        assert methods[method].bounds is False


@pytest.mark.parametrize("method", BOUNDED_METHODS)
def test_bounded_methods_accept_and_respect_box_constraints(method):
    result = BACKEND.solve(
        PROBLEM, method, START, {"max_iter": 100, "bounds": [(0.5, np.inf), (-np.inf, np.inf)]}
    )

    assert result.x_final[0] >= 0.5 - 1e-6
    assert np.isclose(result.x_final[0], 0.5, atol=1e-6)


@pytest.mark.parametrize("method", UNBOUNDED_METHODS)
def test_unbounded_methods_reject_box_constraints(method):
    with pytest.raises(ValueError):
        BACKEND.solve(
            PROBLEM, method, START, {"max_iter": 100, "bounds": [(0.5, np.inf), (-np.inf, np.inf)]}
        )


def test_capabilities_losses_are_empty_for_every_method():
    # minimize has no robust-loss concept; that is a least_squares-specific
    # option (Section 9 item 4), not something any of these methods offer.
    methods = BACKEND.capabilities().methods
    for method in MINIMIZE_METHODS:
        assert methods[method].losses == frozenset()


def test_capabilities_derivative_modes_match_which_methods_support_finite_difference():
    methods = BACKEND.capabilities().methods

    for method in FD_CAPABLE_METHODS:
        assert "finite-difference" in methods[method].derivative_modes
    for method in FD_INCAPABLE_METHODS:
        assert methods[method].derivative_modes == frozenset({"analytic"})


@pytest.mark.parametrize("method", FD_CAPABLE_METHODS)
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


@pytest.mark.parametrize("method", FD_INCAPABLE_METHODS)
def test_finite_difference_derivative_mode_is_rejected_where_scipy_requires_an_analytic_jacobian(
    method,
):
    with pytest.raises(ValueError):
        BACKEND.solve(
            PROBLEM, method, START, {"max_iter": 100, "derivative_mode": "finite-difference"}
        )


def test_solve_falls_back_to_finite_difference_when_problem_has_no_analytic_jacobian():
    problem = dataclasses.replace(PROBLEM, jacobian=None)

    result = BACKEND.solve(problem, "BFGS", START, {"max_iter": 100})

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)
