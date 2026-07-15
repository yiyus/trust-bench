import numpy as np

import trust_bench.backends.scipy_backend as scipy_backend_module
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus, TimingStats
from trust_bench.core.timing import N_REPS, WARMUP
from trust_bench.problems import rosenbrock

BACKEND = SciPyBackend()


def test_solve_populates_real_timing_stats_not_none():
    result = BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert isinstance(result.timing, TimingStats)
    assert result.timing.median > 0.0
    assert result.timing.mad >= 0.0
    assert result.timing.n_reps == N_REPS
    assert result.timing.warmup == WARMUP


def test_solve_pins_the_blas_thread_count_for_the_measurement(monkeypatch):
    # Thread count must be actually pinned for the measurement, not just
    # recorded after the fact (this issue's own acceptance criterion) -
    # threadpool_limits is the standard mechanism for BLAS/NumPy.
    captured = {}
    real_threadpool_limits = scipy_backend_module.threadpool_limits

    def spy(*args, **kwargs):
        captured["limits"] = kwargs.get("limits")
        return real_threadpool_limits(*args, **kwargs)

    monkeypatch.setattr(scipy_backend_module, "threadpool_limits", spy)

    result = BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert captured.get("limits") is not None
    assert result.timing.thread_count == captured["limits"]


def test_solve_still_returns_the_real_solution_despite_the_repeated_measurement_calls():
    # Warm-up and repeated measurement calls solve the same problem
    # several times over; the RunResult returned must still reflect a
    # genuine, converged solve - not, say, a stale warm-up result or a
    # solve whose x0 was mutated by an earlier repetition.
    result = BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED
    assert np.allclose(result.x_final, rosenbrock.PROBLEM.optima[0].x_star, atol=1e-6)
