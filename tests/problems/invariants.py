import numpy as np
from fd_stencils import fd_gradient, fd_hessian, fd_jacobian


def _objective(problem):
    def objective(x):
        r = np.asarray(problem.residual(x), dtype=float)
        return 0.5 * float(r @ r)

    return objective


def assert_parity(problem, jacobian_tol=None, hessian_tol=None):
    """Assert analytic Jacobian/Hessian match finite differences at every
    probe_point (Section 5's analytic-vs-FD parity guard)."""
    jacobian_tol = jacobian_tol or dict(atol=1e-6)
    hessian_tol = hessian_tol or dict(atol=1e-2)
    assert problem.probe_points, f"{problem.id} has no probe_points to check"
    objective = _objective(problem)
    for point in problem.probe_points:
        assert np.allclose(problem.jacobian(point), fd_jacobian(problem.residual, point), **jacobian_tol)
        assert np.allclose(problem.hessian(point), fd_hessian(objective, point), **hessian_tol)


def assert_known_optimum(problem, atol=1e-6):
    """Assert every known optimum has zero gradient and the registered
    cost (Section 5's known-optimum invariant). A kind="scalar" problem
    has no analytic Jacobian by construction (residual() already is the
    scalar cost); its gradient is checked via finite differences
    instead of the residuals-kind J.T@r formula."""
    assert problem.optima, f"{problem.id} has no known optima to check"
    for optimum in problem.optima:
        if problem.kind == "scalar":
            cost_star = float(problem.residual(optimum.x_star))
            grad_star = fd_gradient(problem.residual, optimum.x_star)
        else:
            r_star = np.asarray(problem.residual(optimum.x_star), dtype=float)
            grad_star = problem.jacobian(optimum.x_star).T @ r_star
            cost_star = 0.5 * float(r_star @ r_star)
        assert np.allclose(grad_star, 0.0, atol=atol)
        assert np.isclose(cost_star, optimum.cost_star, atol=atol)
