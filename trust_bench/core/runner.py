import contextlib
import contextvars
import dataclasses

from trust_bench.core.backend import Backend
from trust_bench.core.config import RunConfig
from trust_bench.core.problem import Problem
from trust_bench.core.result import RunResult

_measure_timing = contextvars.ContextVar("measure_timing", default=False)


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


def run(problem: Problem, backend: Backend, method: str, start: str, config: RunConfig) -> RunResult:
    if _measure_timing.get() and not config.measure_timing:
        config = dataclasses.replace(config, measure_timing=True)
    return backend.solve(problem, method, start, config)
