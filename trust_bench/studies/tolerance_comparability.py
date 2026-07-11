from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import CANONICAL_PROBLEMS

METHOD = "lm"
TOLERANCE = 1e-8


def sweep(tolerance=TOLERANCE, max_iter=1000, backends=BACKENDS):
    """RunResult per (problem_id, backend_name): every canonical problem
    solved by "lm" at one identical, explicit RunConfig.tolerance across
    backends, so dist_to_opt/grad_norm_final can be read side by side
    without each backend silently falling back to its own native
    tolerance default (Section 7 of docs/plans/trust-bench.md).

    An equal tolerance value is not expected to yield comparable
    precision: see docs/methodology.md for why the two backends'
    tolerance parameters are not semantically equivalent stopping
    criteria.
    """
    config = RunConfig(max_iter=max_iter, tolerance=tolerance)
    return {
        (problem.id, backend.name): run(problem, backend, METHOD, "standard", config)
        for problem in CANONICAL_PROBLEMS
        for backend in backends
    }
