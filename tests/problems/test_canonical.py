import numpy as np
import pytest
from fd_stencils import fd_hessian, fd_jacobian

from trust_bench.problems import beale, expdec, helical, linear, powell, quadratic, rosenbrock

PROBLEMS = [
    rosenbrock.PROBLEM,
    beale.PROBLEM,
    powell.PROBLEM,
    helical.PROBLEM,
    expdec.PROBLEM,
    quadratic.PROBLEM,
    linear.PROBLEM,
]


def _objective(problem):
    def objective(x):
        r = np.asarray(problem.residual(x), dtype=float)
        return 0.5 * float(r @ r)

    return objective


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_analytic_jacobian_matches_finite_differences_at_every_probe_point(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    for point in problem.probe_points:
        analytic = problem.jacobian(point)
        fd = fd_jacobian(problem.residual, point)
        assert np.allclose(analytic, fd, atol=1e-6)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_analytic_hessian_matches_finite_differences_at_every_probe_point(problem):
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    objective = _objective(problem)
    for point in problem.probe_points:
        analytic = problem.hessian(point)
        fd = fd_hessian(objective, point)
        assert np.allclose(analytic, fd, atol=1e-2)


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_known_optimum_satisfies_gradient_and_cost_invariants(problem):
    assert problem.optima, f"{problem.id} has no known optima to check"
    for optimum in problem.optima:
        r_star = np.asarray(problem.residual(optimum.x_star), dtype=float)
        grad_star = problem.jacobian(optimum.x_star).T @ r_star
        assert np.allclose(grad_star, 0.0, atol=1e-8)
        cost_star = 0.5 * float(r_star @ r_star)
        assert np.isclose(cost_star, optimum.cost_star, atol=1e-8)
