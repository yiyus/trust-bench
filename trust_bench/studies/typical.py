from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import TYPICAL_PROBLEMS

_CONFIG = RunConfig(max_iter=200)


def sweep(problems=TYPICAL_PROBLEMS, backends=BACKENDS):
    """RunResult per (problem_id, method, backend_name): every typical
    problem solved by every method a backend declares support for, from
    its own ordinary "standard" start - not a deliberately-far or
    deliberately-degenerate one, unlike the difficulty-family studies.
    """
    results = {}
    for problem in problems:
        for backend in backends:
            for method in backend.capabilities().methods:
                results[(problem.id, method, backend.name)] = run(
                    problem, backend, method, "standard", _CONFIG
                )
    return results
