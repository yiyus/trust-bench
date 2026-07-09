from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.metrics import basin_rate
from trust_bench.core.runner import run
from trust_bench.problems import beale, expdec, helical, linear, powell, quadratic, rosenbrock

CANONICAL_PROBLEMS = [
    rosenbrock.PROBLEM,
    beale.PROBLEM,
    powell.PROBLEM,
    helical.PROBLEM,
    expdec.PROBLEM,
    quadratic.PROBLEM,
    linear.PROBLEM,
]

_METHOD = "lm"
_CONFIG = RunConfig(max_iter=200)
_BASIN_TOL = 1e-6


def standard_start_results(backends=BACKENDS):
    """Runs the canonical set at its standard start on every backend.

    Regression/parity floor every backend must clear (Section 9 item 1).
    """
    return {
        (problem.id, backend.name): run(problem, backend, _METHOD, "standard", _CONFIG)
        for problem in CANONICAL_PROBLEMS
        for backend in backends
    }


def basin_rates(backends=BACKENDS):
    """Basin-of-attraction rate per problem and backend, across every
    registered start, not only "standard" (Section 9 item 1).
    """
    rates = {}
    for problem in CANONICAL_PROBLEMS:
        for backend in backends:
            distances = [
                run(problem, backend, _METHOD, start, _CONFIG).dist_to_opt
                for start in problem.starts
            ]
            rates[(problem.id, backend.name)] = basin_rate(distances, _BASIN_TOL)
    return rates
