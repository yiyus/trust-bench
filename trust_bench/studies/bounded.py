import dataclasses

import numpy as np

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems import quadratic

METHODS = ["trf", "dogbox"]

# quadratic.PROBLEM's unconstrained optimum is (0, 0); a lower bound of
# 0.5 on x1 makes that infeasible, so the constrained optimum sits
# exactly on the boundary.
_ACTIVE_BOUNDS = ([0.5, -10.0], [10.0, 10.0])
_INACTIVE_BOUNDS = ([-10.0, -10.0], [10.0, 10.0])

SCENARIOS = {
    "inactive": dict(bounds=_INACTIVE_BOUNDS, start=[1.0, -1.0]),
    "active_at_boundary": dict(bounds=_ACTIVE_BOUNDS, start=[1.0, -1.0]),
    "infeasible_start": dict(bounds=_ACTIVE_BOUNDS, start=[-5.0, -5.0]),
}


def sweep(scenarios=SCENARIOS, methods=METHODS, backends=BACKENDS):
    """RunResult, or the raised ValueError, per (scenario, method,
    backend_name). A start outside its scenario's bounds is expected to
    raise: SciPy's least_squares validates x0 against bounds itself and
    does not silently project an infeasible start.
    """
    outcomes = {}
    for name, scenario in scenarios.items():
        problem = dataclasses.replace(
            quadratic.PROBLEM, starts={"scenario": np.array(scenario["start"])}
        )
        config = RunConfig(max_iter=200, bounds=scenario["bounds"])
        for backend in backends:
            for method in methods:
                try:
                    outcomes[(name, method, backend.name)] = run(
                        problem, backend, method, "scenario", config
                    )
                except ValueError as error:
                    outcomes[(name, method, backend.name)] = error
    return outcomes
