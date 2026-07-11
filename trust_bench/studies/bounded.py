import dataclasses

import numpy as np

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import quadratic

# Mirrors scipy_backend.py's own _LEAST_SQUARES_METHODS and
# reporting/capability_matrix.py's _LEAST_SQUARES_BOUNDS_PROBE split:
# these methods (and trust-apl's "lm") take a (lower, upper) array pair,
# while scipy's minimize-family methods take a per-dimension (low, high)
# pair list instead.
_LEAST_SQUARES_METHODS = frozenset({"lm", "trf", "dogbox"})

# quadratic.PROBLEM's unconstrained optimum is (0, 0); a lower bound of
# 0.5 on x1 makes that infeasible, so the constrained optimum sits
# exactly on the boundary.
_LEAST_SQUARES_ACTIVE_BOUNDS = ([0.5, -10.0], [10.0, 10.0])
_LEAST_SQUARES_INACTIVE_BOUNDS = ([-10.0, -10.0], [10.0, 10.0])
_MINIMIZE_ACTIVE_BOUNDS = [(0.5, 10.0), (-10.0, 10.0)]
_MINIMIZE_INACTIVE_BOUNDS = [(-10.0, 10.0), (-10.0, 10.0)]

SCENARIOS = {
    "inactive": dict(active=False, start=[1.0, -1.0]),
    "active_at_boundary": dict(active=True, start=[1.0, -1.0]),
    "infeasible_start": dict(active=True, start=[-5.0, -5.0]),
}


def _bounds_for(method, active):
    if method in _LEAST_SQUARES_METHODS:
        return _LEAST_SQUARES_ACTIVE_BOUNDS if active else _LEAST_SQUARES_INACTIVE_BOUNDS
    return _MINIMIZE_ACTIVE_BOUNDS if active else _MINIMIZE_INACTIVE_BOUNDS


def sweep(scenarios=SCENARIOS, backends=BACKENDS):
    """RunResult, or the raised ValueError, per (scenario, method,
    backend_name), for every method a backend declares bounds support
    for, rather than a fixed method list: trust-apl's Coleman-Li-scaled
    "lm" is exercised here, alongside every one of scipy's own
    bounds-capable methods.

    A start outside its scenario's bounds is not universally rejected:
    scipy's least-squares methods validate x0 against bounds and raise,
    while scipy's minimize-family methods and trust-apl's "lm" project
    it into the box and converge regardless - both are genuine method
    behaviours, so this records whichever one actually happens.
    """
    outcomes = {}
    for name, scenario in scenarios.items():
        problem = dataclasses.replace(
            quadratic.PROBLEM, starts={"scenario": np.array(scenario["start"])}
        )
        for backend in backends:
            for method, caps in backend.capabilities().methods.items():
                if not caps.bounds:
                    continue
                config = RunConfig(max_iter=200, bounds=_bounds_for(method, scenario["active"]))
                try:
                    outcomes[(name, method, backend.name)] = run(
                        problem, backend, method, "scenario", config
                    )
                except ValueError as error:
                    outcomes[(name, method, backend.name)] = error
    return outcomes
