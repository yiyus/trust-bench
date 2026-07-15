import shutil
import time

import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus, TimingStats
from trust_bench.core.timing import N_REPS, WARMUP
from trust_bench.problems import rosenbrock

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()


def test_timing_defaults_to_none():
    # measure_timing defaults to False: a plain solve costs exactly one
    # request/response round trip, unchanged from before RunResult.timing
    # existed - repeating every solve() call by WARMUP+N_REPS regardless
    # of problem cost is what real report generation wants, not an
    # ordinary correctness check (confirmed directly to matter: a single
    # already-slow test - dimensionality(n=1000)/trust-exact - went from
    # ~4s to ~23s when this was unconditional).
    result = BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))

    assert result.timing is None


def test_solve_populates_real_timing_stats_when_measurement_is_requested():
    result = BACKEND.solve(
        rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200, measure_timing=True)
    )

    assert isinstance(result.timing, TimingStats)
    assert result.timing.median > 0.0
    assert result.timing.mad >= 0.0
    assert result.timing.n_reps == N_REPS
    assert result.timing.warmup == WARMUP


def test_reported_timing_excludes_the_persistent_sessions_own_ipc_overhead():
    # The whole point of measuring inside the APL script (⎕AI around the
    # Min call in solve.dyalog) rather than wall-clocking the round trip
    # from Python: a naive external wrap would include JSON/IPC transfer
    # time as if it were solve time. Confirmed directly by comparing the
    # reported median against a real external wall-clock measurement of
    # the same warmed-up session, which necessarily also includes that
    # transfer overhead - the reported value must be no larger.
    BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200))  # warm the session

    config = RunConfig(max_iter=200, measure_timing=True)
    start = time.perf_counter()
    result = BACKEND.solve(rosenbrock.PROBLEM, "lm", "standard", config)
    external_elapsed = time.perf_counter() - start

    assert result.timing.median <= external_elapsed


def test_solve_still_returns_the_real_solution_despite_the_repeated_measurement_calls():
    result = BACKEND.solve(
        rosenbrock.PROBLEM, "lm", "standard", RunConfig(max_iter=200, measure_timing=True)
    )

    assert result.status is RunStatus.CONVERGED
    assert result.x_final == pytest.approx(rosenbrock.PROBLEM.optima[0].x_star.tolist(), abs=1e-4)
