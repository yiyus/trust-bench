import dataclasses

import numpy as np
import pytest

import trust_bench.backends.scipy_backend as scipy_backend_module
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.core.timing import N_REPS, WARMUP
from trust_bench.problems import quadratic

BACKEND = SciPyBackend()
PROBLEM = quadratic.PROBLEM
START = "standard"
LEAST_SQUARES_METHODS = ["lm", "trf", "dogbox"]


def _spy_on_least_squares(monkeypatch):
    captured = {}
    real_least_squares = scipy_backend_module.least_squares

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_least_squares(*args, **kwargs)

    monkeypatch.setattr(scipy_backend_module, "least_squares", spy)
    return captured


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_method_solves_the_trivial_quadratic_to_the_known_optimum(method):
    result = BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100))

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, PROBLEM.optima[0].x_star, atol=1e-6)


def test_message_carries_scipys_own_termination_explanation():
    # least_squares already computes a human-readable reason for every
    # termination, converged or not; surfacing it costs nothing and is
    # what actually distinguishes a MAX_ITER row from an opaque one.
    result = BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=100))

    assert result.message
    assert isinstance(result.message, str)


def test_capabilities_bounds_flag_matches_which_methods_accept_bounds():
    methods = BACKEND.capabilities().methods

    assert methods["lm"].bounds is False
    assert methods["trf"].bounds is True
    assert methods["dogbox"].bounds is True


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_bounded_methods_accept_and_respect_box_constraints(method):
    result = BACKEND.solve(
        PROBLEM,
        method,
        START,
        RunConfig(max_iter=100, bounds=([0.5, -np.inf], [np.inf, np.inf])),
    )

    assert result.x_final[0] >= 0.5 - 1e-9
    assert np.isclose(result.x_final[0], 0.5, atol=1e-6)


def test_lm_rejects_box_constraints():
    with pytest.raises(ValueError):
        BACKEND.solve(
            PROBLEM,
            "lm",
            START,
            RunConfig(max_iter=100, bounds=([0.5, -np.inf], [np.inf, np.inf])),
        )


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
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


def test_solve_falls_back_to_finite_difference_when_problem_has_no_analytic_jacobian():
    problem = dataclasses.replace(PROBLEM, jacobian=None)

    result = BACKEND.solve(problem, "lm", START, RunConfig(max_iter=100))

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
    # happens if solve() actually forwards config.loss to
    # least_squares(loss=...), so a solve() that silently ignores the field
    # would not raise here.
    with pytest.raises(ValueError):
        BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100, loss="not-a-real-loss"))


def test_lm_rejects_a_non_linear_loss():
    with pytest.raises(ValueError):
        BACKEND.solve(PROBLEM, "lm", START, RunConfig(max_iter=100, loss="huber"))


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_tolerance_maps_to_ftol_xtol_and_gtol(method, monkeypatch):
    # least_squares exposes all three stopping criteria simultaneously
    # (Section 7 of docs/plans/trust-bench.md); a single intent-level
    # tolerance is applied to all three rather than leaving two of them at
    # scipy's own defaults.
    captured = _spy_on_least_squares(monkeypatch)

    BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100, tolerance=1e-5))

    assert captured["ftol"] == 1e-5
    assert captured["xtol"] == 1e-5
    assert captured["gtol"] == 1e-5


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_no_tolerance_config_leaves_scipys_own_defaults_in_place(method, monkeypatch):
    captured = _spy_on_least_squares(monkeypatch)

    BACKEND.solve(PROBLEM, method, START, RunConfig(max_iter=100))

    assert "ftol" not in captured
    assert "xtol" not in captured
    assert "gtol" not in captured


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_n_feval_counts_every_residual_call_including_finite_difference_jacobian_steps(method):
    # Verified directly against scipy: result.nfev only counts calls the
    # outer algorithm makes itself, not the extra residual calls scipy's
    # own finite-difference Jacobian estimation makes internally, which
    # for a 2-variable problem is most of the true call count (e.g. 7
    # real calls vs. 2 reported, for every method here).
    calls = []

    def counting_residual(x):
        calls.append(x)
        return PROBLEM.residual(x)

    problem = dataclasses.replace(PROBLEM, residual=counting_residual)

    result = BACKEND.solve(
        problem, method, START, RunConfig(max_iter=100, derivative_mode="finite-difference")
    )

    # solve() now performs WARMUP+N_REPS identical, deterministic solves
    # (docs/prs/138.md's timing instrumentation) - the spy counts every
    # one of them, while n_feval reports only the last repetition's own
    # count.
    assert result.n_feval == len(calls) / (WARMUP + N_REPS)


@pytest.mark.parametrize("method", LEAST_SQUARES_METHODS)
def test_n_feval_does_not_count_an_extra_call_made_only_to_report_grad_norm_final(method):
    # A second, independent undercount: solve() used to recompute
    # problem.residual(result.x) after least_squares returned, purely to
    # derive grad_norm_final, adding one genuine extra call that scipy's
    # own nfev never counted either. Verified directly: real calls are
    # exactly reported nfev + 1 in analytic mode, for every method, with
    # the un-fixed implementation.
    calls = []

    def counting_residual(x):
        calls.append(x)
        return PROBLEM.residual(x)

    problem = dataclasses.replace(PROBLEM, residual=counting_residual)

    result = BACKEND.solve(problem, method, START, RunConfig(max_iter=100))

    # See the finite-difference case above: WARMUP+N_REPS identical
    # solves, spy counts every one, n_feval reports only the last.
    assert result.n_feval == len(calls) / (WARMUP + N_REPS)
