from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.problems import CANONICAL_PROBLEMS
from trust_bench.studies.derivative_source import DERIVATIVE_MODES, METHODS, sweep

_PRECISION_TOL = 1e-6
# Moré-Garbow-Hillstrom #13 ("Powell singular function"): its Jacobian
# is singular at the true optimum by construction, so trust-region
# methods that need a well-conditioned Jacobian near the optimum
# plateau well short of machine precision on it, regardless of
# tolerance. Verified directly: trf/dogbox plateau at ~6e-4 on this
# problem in both derivative modes, even with RunConfig.tolerance
# tightened to 1e-12; lm does not share this (reaches ~1e-9 or better).
_PRECISION_TOL_OVERRIDES = {"powell": 1e-3}


def _precision_tolerance(problem_id):
    return _PRECISION_TOL_OVERRIDES.get(problem_id, _PRECISION_TOL)


def test_sweep_covers_every_problem_method_mode_and_backend():
    results = sweep()

    assert len(results) == len(CANONICAL_PROBLEMS) * len(METHODS) * len(DERIVATIVE_MODES) * len(
        BACKENDS
    )


def test_both_derivative_modes_converge_precisely_for_every_problem_and_method():
    results = sweep()

    for (problem_id, method, mode, backend_name), result in results.items():
        label = f"{problem_id}/{method}/{mode}/{backend_name}"
        assert result.status is RunStatus.CONVERGED, label
        assert result.dist_to_opt < _precision_tolerance(problem_id), label


def test_finite_difference_needs_more_function_evaluations_than_analytic():
    # Verified directly against scipy, once n_feval was fixed to count
    # every residual call (including the ones spent estimating the
    # Jacobian): every canonical problem, every method, needs strictly
    # more evaluations with a finite-difference Jacobian than with the
    # analytic one, since each Jacobian estimate costs extra residual
    # calls the analytic Jacobian never needs.
    results = sweep()

    for problem in CANONICAL_PROBLEMS:
        for method in METHODS:
            for backend in BACKENDS:
                analytic = results[(problem.id, method, "analytic", backend.name)]
                finite_difference = results[(problem.id, method, "finite-difference", backend.name)]
                label = f"{problem.id}/{method}/{backend.name}"
                assert finite_difference.n_feval > analytic.n_feval, label
