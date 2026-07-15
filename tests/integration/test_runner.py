from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import capture
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.core.runner import measuring_timing, run
from trust_bench.problems import rosenbrock

SCIPY = SciPyBackend()


def test_run_on_rosenbrock_returns_a_run_result_with_intrinsic_metrics_populated():
    result = run(rosenbrock.PROBLEM, SCIPY, "lm", "standard", RunConfig(max_iter=200))

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.CONVERGED
    assert result.dist_to_opt is not None
    assert result.cost_gap is not None
    assert result.grad_norm_final is not None
    assert result.dist_to_opt < 1e-6
    assert result.cost_gap < 1e-6


def test_run_reports_trace_as_unavailable_rather_than_omitting_it_when_the_backend_lacks_one():
    result = run(rosenbrock.PROBLEM, SCIPY, "lm", "standard", RunConfig(max_iter=200))

    assert hasattr(result, "trace")
    assert result.trace is None


class _StubBackendWithTrace(Backend):
    """Returns a fixed RunResult with a populated trace. Proves run() is a
    transparent pass-through that never drops or overwrites a backend's
    trace; no production backend captures per-iterate traces yet, so this
    is the only way to exercise that branch.
    """

    name = "stub-with-trace"

    def capabilities(self):
        return Capabilities(
            methods={
                "gd": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                )
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=[1.0, 1.0],
            cost_final=0.0,
            dist_to_opt=0.0,
            cost_gap=0.0,
            grad_norm_final=0.0,
            status=RunStatus.CONVERGED,
            n_iter=1,
            n_feval=1,
            n_jeval=1,
            n_heval=0,
            trace=[[0.0, 0.0], [1.0, 1.0]],
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha="test",
            timestamp="2026-01-01T00:00:00Z",
        )


def test_run_passes_through_the_backends_trace_when_it_captures_one():
    result = run(rosenbrock.PROBLEM, _StubBackendWithTrace(), "gd", "standard", RunConfig())

    assert result.trace == [[0.0, 0.0], [1.0, 1.0]]


def test_run_does_not_force_measure_timing_outside_the_measuring_timing_context():
    result = run(rosenbrock.PROBLEM, _StubBackendWithTrace(), "gd", "standard", RunConfig())

    assert result.config.measure_timing is False


def test_measuring_timing_forces_measure_timing_on_for_every_run_call_inside_it():
    # A study's own sweep() never sets measure_timing itself (it would
    # slow down every test that calls the same function directly for
    # correctness, not performance) - trust-bench report's real run is
    # the one place that needs it, via this context manager wrapping
    # every study it runs, not a parameter threaded through by hand into
    # each study's own RunConfig construction.
    with measuring_timing():
        result = run(rosenbrock.PROBLEM, _StubBackendWithTrace(), "gd", "standard", RunConfig())

    assert result.config.measure_timing is True


def test_measuring_timing_overrides_an_explicit_false_from_the_caller():
    with measuring_timing():
        result = run(
            rosenbrock.PROBLEM,
            _StubBackendWithTrace(),
            "gd",
            "standard",
            RunConfig(measure_timing=False),
        )

    assert result.config.measure_timing is True
