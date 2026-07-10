import time

import pytest

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.core.runner import run
from trust_bench.problems.families import dimensionality
from trust_bench.studies.dimensionality import DENSE_METHODS, MATRIX_FREE_METHODS, METHODS, N_VALUES, sweep

_LARGEST_N = 1000
_SLOW_METHOD_TIME_RATIO = 5


@pytest.mark.slow
def test_sweep_covers_every_n_method_and_backend():
    # A tiny max_iter here: this test checks structure, not
    # convergence, and a dense method's per-step cost at n=1000 (the
    # whole point of this study) is high enough that a full
    # convergence budget would make this one structural check dominate
    # the suite's runtime.
    results = sweep(max_iter=3)

    assert len(results) == len(N_VALUES) * len(METHODS) * len(BACKENDS)


@pytest.mark.slow
def test_matrix_free_methods_converge_precisely_at_every_dimension():
    results = sweep(methods=MATRIX_FREE_METHODS)

    for (n, method, backend_name), result in results.items():
        assert result.status is RunStatus.CONVERGED, f"n={n} method={method} on {backend_name}"
        assert result.dist_to_opt < 1e-3, f"n={n} method={method} on {backend_name}"


def test_bfgs_fails_to_converge_at_large_dimension():
    # A dense quasi-Newton method builds and maintains a full n x n
    # approximate Hessian with no curvature shortcut, and stops making
    # real progress well before the trivial generalised-Rosenbrock
    # optimum, at a dimension the matrix-free methods reach easily.
    # n=100 (not the largest, n=1000) keeps this test fast: the failure
    # is already unmistakable there.
    n = 100
    problem = dimensionality.make(n)
    for backend in BACKENDS:
        result = run(problem, backend, "BFGS", "standard", RunConfig(max_iter=200))
        assert result.dist_to_opt > 0.1, f"n={n} on {backend.name}"


@pytest.mark.slow
def test_dense_hessian_methods_cost_far_more_per_step_at_large_dimension():
    # The study's core claim, measured directly: a handful of steps at
    # n=1000 costs much more wall-clock time for a dense-Hessian method
    # than for L-BFGS-B, which never forms an n x n matrix at all. A
    # small max_iter isolates per-step cost from iteration count:
    # trust-exact's eigendecomposition-based subproblem solve dominates
    # its per-step cost regardless of how many steps are taken.
    n = _LARGEST_N
    problem = dimensionality.make(n)
    config = RunConfig(max_iter=2)

    for backend in BACKENDS:
        start = time.perf_counter()
        run(problem, backend, "L-BFGS-B", "standard", config)
        reference_time = time.perf_counter() - start

        for method in DENSE_METHODS:
            start = time.perf_counter()
            run(problem, backend, method, "standard", config)
            elapsed = time.perf_counter() - start
            assert elapsed > _SLOW_METHOD_TIME_RATIO * reference_time, f"{method} on {backend.name}"
