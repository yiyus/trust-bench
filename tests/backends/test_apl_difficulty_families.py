import dataclasses
import shutil

import numpy as np
import pytest

from trust_bench.backends.apl_backend import APLBackend
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems.families import ill_conditioned, large_residual, outliers, scaling

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

BACKEND = APLBackend()
START = "standard"

# One representative parameter value per family, matching
# tests/problems/test_families.py's own parity-check values.
DIFFICULTY_FAMILY_PROBLEMS = [
    scaling.make(s=10.0),
    ill_conditioned.make(kappa=100.0),
    large_residual.make(rho=10.0),
    outliers.make(fraction=0.3),
]


@pytest.mark.parametrize("problem", DIFFICULTY_FAMILY_PROBLEMS, ids=lambda p: p.id)
def test_solve_converges_to_the_known_optimum_instead_of_returning_error(problem):
    result = BACKEND.solve(problem, "lm", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.CONVERGED, result.status
    assert np.allclose(result.x_final, problem.optima[0].x_star, atol=1e-3)


def test_solve_reports_unknown_problem_id_for_an_unrecognised_parametrised_family():
    problem = dataclasses.replace(scaling.make(s=10.0), id="not_a_family(x=1.0)")

    result = BACKEND.solve(problem, "lm", START, RunConfig(max_iter=200))

    assert result.status is RunStatus.ERROR
