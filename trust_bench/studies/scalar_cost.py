from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import SCALAR_PROBLEMS

METHODS = ["BFGS", "L-BFGS-B"]
_CONFIG = RunConfig(max_iter=200)


def sweep(problems=SCALAR_PROBLEMS, methods=METHODS, backends=BACKENDS):
    """RunResult per (problem_id, method, backend_name): BFGS/L-BFGS-B
    solving genuine scalar-cost problems with no residual/Jacobian to
    exploit. trust-apl is not wired in for these problems - its own
    BFGS engine always derives cost/gradient from a residual-and-
    Jacobian-bearing evaluator internally (backends_ext/apl/solve.dyalog),
    so it reports ERROR ("Unknown problem_id") for both, since no
    APL-side evaluator exists for a Jacobian-free scalar objective.
    Skips a (method, backend) pair the backend does not support.
    """
    results = {}
    for problem in problems:
        for backend in backends:
            supported = backend.capabilities().methods
            for method in methods:
                if method not in supported:
                    continue
                results[(problem.id, method, backend.name)] = run(problem, backend, method, "standard", _CONFIG)
    return results
