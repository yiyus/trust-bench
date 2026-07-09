from trust_bench.core.backend import Backend
from trust_bench.core.config import RunConfig
from trust_bench.core.problem import Problem
from trust_bench.core.result import RunResult


def run(problem: Problem, backend: Backend, method: str, start: str, config: RunConfig) -> RunResult:
    return backend.solve(problem, method, start, config)
