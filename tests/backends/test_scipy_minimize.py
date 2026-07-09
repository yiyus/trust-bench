import dataclasses

import numpy as np
import pytest

import trust_bench.backends.scipy_backend as scipy_backend_module
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.config import RunConfig
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
# Verified via scipy.optimize.show_options(solver="minimize", method=...):
# the subset of {ftol, xtol, gtol} each method's `options` dict accepts.
TOLERANCE_PARAMS = {
    "BFGS": frozenset({"gtol"}),
    "L-BFGS-B": frozenset({"ftol", "gtol"}),
    "Newton-CG": frozenset({"xtol"}),
    "trust-exact": frozenset({"gtol"}),
    "trust-constr": frozenset({"gtol", "xtol"}),
}


def _spy_on_minimize(monkeypatch):
    captured = {}
    real_minimize = scipy_backend_module.minimize

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(scipy_backend_module, "minimize", spy)
    return captured


@pytest.mark.parametrize("method", MINIMIZE_METHODS)
def test_method_solves_the_trivial_quadratic_to_the_known_optimum(method):
    result = BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100))

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
        PROBLEM,
        method,
        START,
        RunConfig(max_iter=100, bounds=[(0.5, np.inf), (-np.inf, np.inf)]),
    )

    # trust-constr's default convergence tolerance is looser than
    # L-BFGS-B's for this trivial problem (verified: ~6e-5 vs ~1e-21
    # error), so this atol is calibrated to trust-constr, not L-BFGS-B.
    assert result.x_final[0] >= 0.5 - 1e-6
    assert np.isclose(result.x_final[0], 0.5, atol=1e-3)


@pytest.mark.parametrize("method", UNBOUNDED_METHODS)
def test_unbounded_methods_reject_box_constraints(method):
    with pytest.raises(ValueError):
        BACKEND.solve(
            PROBLEM,
            method,
            START,
            RunConfig(max_iter=100, bounds=[(0.5, np.inf), (-np.inf, np.inf)]),
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
        problem, method, START, RunConfig(max_iter=100, derivative_mode="finite-difference")
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
            PROBLEM, method, START, RunConfig(max_iter=100, derivative_mode="finite-difference")
        )


def test_solve_falls_back_to_finite_difference_when_problem_has_no_analytic_jacobian():
    problem = dataclasses.replace(PROBLEM, jacobian=None)

    result = BACKEND.solve(problem, "BFGS", START, RunConfig(max_iter=100))

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_fd_capable_hessian_method_still_solves_when_problem_has_no_analytic_hessian():
    # trust-constr is the only Hessian-using method whose derivative_modes
    # includes finite-difference (verified in test_capabilities_derivative_
    # modes_match_which_methods_support_finite_difference), so it is the
    # only one that can fall back to a Hessian update strategy here.
    problem = dataclasses.replace(PROBLEM, hessian=None)

    result = BACKEND.solve(problem, "trust-constr", START, RunConfig(max_iter=100))

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


@pytest.mark.parametrize("method", FD_INCAPABLE_METHODS)
def test_analytic_only_hessian_methods_reject_a_problem_with_no_analytic_hessian(method):
    # Newton-CG and trust-exact both declare derivative_modes={"analytic"};
    # a problem with no analytic Hessian is, for these two, the same kind
    # of unsupported finite-difference request as no analytic Jacobian,
    # even though scipy itself would let Newton-CG silently fall back to a
    # different (Hessian-free) mechanism if asked directly.
    problem = dataclasses.replace(PROBLEM, hessian=None)

    with pytest.raises(ValueError):
        BACKEND.solve(problem, method, START, RunConfig(max_iter=100))


@pytest.mark.parametrize("method", MINIMIZE_METHODS)
def test_tolerance_maps_to_the_correct_native_options_per_method(method, monkeypatch):
    captured = _spy_on_minimize(monkeypatch)

    BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100, tolerance=1e-5))

    options = captured["options"]
    for param in TOLERANCE_PARAMS[method]:
        assert options[param] == 1e-5
    # scipy itself rejects an option a method doesn't recognise, but that
    # exercises scipy's own validation, not this mapping; assert directly
    # that no other tolerance-shaped key was set.
    assert set(options) & {"ftol", "xtol", "gtol"} == TOLERANCE_PARAMS[method]


@pytest.mark.parametrize("method", MINIMIZE_METHODS)
def test_no_tolerance_config_leaves_scipys_own_defaults_in_place(method, monkeypatch):
    captured = _spy_on_minimize(monkeypatch)

    BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100))

    options = captured.get("options") or {}
    assert set(options) & {"ftol", "xtol", "gtol"} == set()
