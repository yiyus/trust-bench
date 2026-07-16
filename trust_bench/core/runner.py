import contextlib
import contextvars
import dataclasses

from trust_bench.core.backend import Backend
from trust_bench.core.config import RunConfig
from trust_bench.core.problem import Problem
from trust_bench.core.result import RunResult
from trust_bench.core.storage import append

_measure_timing = contextvars.ContextVar("measure_timing", default=False)
_recording_path = contextvars.ContextVar("recording_path", default=None)


@contextlib.contextmanager
def measuring_timing():
    """Every run() call made while this context is active gets
    RunConfig.measure_timing forced on, regardless of what the
    caller's own RunConfig already set. trust-bench report's own entry
    point into real TimingStats: every study already funnels through
    run(), so this is the one place that needs to know about a real
    report run, not a per-study parameter threaded through by hand into
    each one's own sweep()/RunConfig construction - which would also
    slow down every test that calls those same functions directly for
    correctness checks, not performance measurement.
    """
    token = _measure_timing.set(True)
    try:
        yield
    finally:
        _measure_timing.reset(token)


@contextlib.contextmanager
def recording_results(path):
    """Every run() call made while this context is active has its
    RunResult appended to `path` (Section 8: results are appended to
    results/*.jsonl, never overwritten). `path=None` is a no-op, so a
    caller that wants persistence disabled can pass it through
    unconditionally rather than branching around the context manager
    itself.
    """
    token = _recording_path.set(path)
    try:
        yield
    finally:
        _recording_path.reset(token)


def run(problem: Problem, backend: Backend, method: str, start: str, config: RunConfig) -> RunResult:
    if _measure_timing.get() and not config.measure_timing:
        config = dataclasses.replace(config, measure_timing=True)
    result = backend.solve(problem, method, start, config)
    path = _recording_path.get()
    if path is not None:
        append(result, path)
    return result
