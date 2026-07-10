from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import CANONICAL_PROBLEMS

METHODS = ["lm", "trf", "dogbox"]
DERIVATIVE_MODES = ["analytic", "finite-difference"]


def sweep(problems=CANONICAL_PROBLEMS, methods=METHODS, derivative_modes=DERIVATIVE_MODES, backends=BACKENDS):
    """RunResult per (problem_id, method, derivative_mode, backend_name):
    evaluation count and precision, analytic vs finite-difference
    Jacobian, for the same problem set.
    """
    results = {}
    for problem in problems:
        for backend in backends:
            for method in methods:
                for mode in derivative_modes:
                    config = RunConfig(max_iter=200, derivative_mode=mode)
                    results[(problem.id, method, mode, backend.name)] = run(
                        problem, backend, method, "standard", config
                    )
    return results
