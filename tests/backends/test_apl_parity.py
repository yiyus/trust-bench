import shutil

import numpy as np
import pytest

from trust_bench.backends.apl_backend import evaluate_problem
from trust_bench.problems import CANONICAL_PROBLEMS, TYPICAL_PROBLEMS
from trust_bench.problems.families import dimensionality, ill_conditioned, large_residual, outliers, scaling

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed"),
]

# One representative parameter value per difficulty family. scaling,
# large_residual and outliers match tests/problems/test_families.py's
# own parity-check values; ill_conditioned's kappa=100.0 matches
# ill_conditioning.py's own study KAPPAS list instead (that file's
# parity checks use 1e3, not 1e2); dimensionality's n=10 matches
# tests/problems/test_families.py's own DIMENSIONALITY_NS_PARITY.
DIFFICULTY_FAMILY_PROBLEMS = [
    scaling.make(s=10.0),
    ill_conditioned.make(kappa=100.0),
    large_residual.make(rho=10.0),
    outliers.make(fraction=0.3),
    dimensionality.make(n=10),
]


@pytest.mark.parametrize("problem", CANONICAL_PROBLEMS, ids=lambda p: p.id)
def test_residual_jacobian_and_hessian_match_the_python_reference_at_every_probe_point(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    for point in problem.probe_points:
        residual, jacobian, hessian = evaluate_problem(problem.id, point)
        assert np.allclose(residual, problem.residual(point), atol=1e-9)
        assert np.allclose(jacobian, problem.jacobian(point), atol=1e-9)
        assert np.allclose(hessian, problem.hessian(point), atol=1e-6)


@pytest.mark.parametrize("problem", DIFFICULTY_FAMILY_PROBLEMS, ids=lambda p: p.id)
def test_difficulty_family_residual_jacobian_and_hessian_match_the_python_reference(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    for point in problem.probe_points:
        residual, jacobian, hessian = evaluate_problem(problem.id, point)
        assert np.allclose(residual, problem.residual(point), atol=1e-9)
        assert np.allclose(jacobian, problem.jacobian(point), atol=1e-9)
        assert np.allclose(hessian, problem.hessian(point), atol=1e-6)


@pytest.mark.parametrize("problem", TYPICAL_PROBLEMS, ids=lambda p: p.id)
def test_typical_problem_residual_jacobian_and_hessian_match_the_python_reference(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    for point in problem.probe_points:
        residual, jacobian, hessian = evaluate_problem(problem.id, point)
        assert np.allclose(residual, problem.residual(point), atol=1e-6)
        assert np.allclose(jacobian, problem.jacobian(point), atol=1e-6)
        assert np.allclose(hessian, problem.hessian(point), atol=1e-3)
