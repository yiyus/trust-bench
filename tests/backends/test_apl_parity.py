import shutil

import numpy as np
import pytest

from trust_bench.backends.apl_backend import evaluate_problem
from trust_bench.problems import CANONICAL_PROBLEMS

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]


@pytest.mark.parametrize("problem", CANONICAL_PROBLEMS, ids=lambda p: p.id)
def test_residual_jacobian_and_hessian_match_the_python_reference_at_every_probe_point(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    for point in problem.probe_points:
        residual, jacobian, hessian = evaluate_problem(problem.id, point)
        assert np.allclose(residual, problem.residual(point), atol=1e-9)
        assert np.allclose(jacobian, problem.jacobian(point), atol=1e-9)
        assert np.allclose(hessian, problem.hessian(point), atol=1e-6)
