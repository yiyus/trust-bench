import statistics
import time

import pytest

from trust_bench.backends import BACKENDS
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.backend import Backend, Capabilities
from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import capture
from trust_bench.core.result import RunStatus
from trust_bench.core.runner import run
from trust_bench.problems.families import dimensionality
from trust_bench.studies.dimensionality import DENSE_METHODS, MATRIX_FREE_METHODS, METHODS, N_VALUES, sweep

_LARGEST_N = 1000
# A single wall-clock measurement of a ~0.1-0.2s reference call is noisy
# enough that its ratio against a several-second dense-method call
# occasionally dips close to a 5x threshold under system load (observed
# directly: as low as 5.07x from one bad reference measurement). The
# median of a few repeats stabilises the reference time; even so,
# measured over 15 repeated trials the minimum observed ratio was
# ~5.8x, so 3x leaves a comfortable margin without weakening the
# claim ("costs far more per step") this test makes.
_SLOW_METHOD_TIME_RATIO = 3
_TIMING_REPS = 3


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


def _median_elapsed(problem, backend, method, config, reps=_TIMING_REPS):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        run(problem, backend, method, "standard", config)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


@pytest.mark.slow
def test_dense_hessian_methods_cost_far_more_per_step_at_large_dimension():
    # The study's core claim, measured directly: a handful of steps at
    # n=1000 costs much more wall-clock time for a dense-Hessian method
    # than for L-BFGS-B, which never forms an n x n matrix at all. A
    # small max_iter isolates per-step cost from iteration count:
    # trust-exact's eigendecomposition-based subproblem solve dominates
    # its per-step cost regardless of how many steps are taken. The
    # median of a few repeats (see _TIMING_REPS) stabilises both sides
    # of the comparison against transient system-load noise.
    n = _LARGEST_N
    problem = dimensionality.make(n)
    config = RunConfig(max_iter=2)

    for backend in BACKENDS:
        reference_time = _median_elapsed(problem, backend, "L-BFGS-B", config)

        for method in DENSE_METHODS:
            elapsed = _median_elapsed(problem, backend, method, config)
            assert elapsed > _SLOW_METHOD_TIME_RATIO * reference_time, f"{method} on {backend.name}"


class _BFGSOnlyBackend(Backend):
    """Wraps SciPyBackend but declares only BFGS, to prove sweep() skips
    a (method, backend) pair the backend does not support rather than
    raising the "no method" ValueError Backend.solve itself would.
    """

    name = "bfgs-only"

    def __init__(self):
        self._scipy = SciPyBackend()

    def capabilities(self):
        methods = self._scipy.capabilities().methods
        return Capabilities(methods={"BFGS": methods["BFGS"]})

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        return self._scipy.solve(problem, method, start, config)


def test_sweep_skips_a_method_a_backend_does_not_support():
    results = sweep(n_values=[10], backends=[_BFGSOnlyBackend()])

    assert set(results) == {(10, "BFGS", "bfgs-only")}
