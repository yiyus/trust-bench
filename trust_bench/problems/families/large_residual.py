import numpy as np
from scipy.optimize import minimize

from trust_bench.core.problem import Optimum, Problem

M = 40
T = np.linspace(0.0, 1.0, M)
A0, K0 = 2.0, 1.3
D = T - 0.5
_X0 = np.array([1.6, 1.0])


def make(rho):
    """A two-parameter exponential-decay fit with an irreducible residual
    that grows with rho: the model cannot represent the added rho*(t-0.5)
    term, so the optimum shifts away from (A0, K0) with no closed form.
    """
    y = A0 * np.exp(K0 * T) + rho * D

    def residual(x):
        a, k = x
        return a * np.exp(k * T) - y

    def jacobian(x):
        a, k = x
        e = np.exp(k * T)
        return np.column_stack([e, a * T * e])

    def hessian(x):
        a, k = x
        e = np.exp(k * T)
        r = residual(x)
        j = jacobian(x)
        h = j.T @ j
        s = np.zeros((2, 2))
        s[0, 1] = s[1, 0] = np.sum(r * T * e)
        s[1, 1] = np.sum(r * a * T**2 * e)
        return h + s

    def objective(x):
        r = residual(x)
        return 0.5 * float(r @ r)

    def grad(x):
        return jacobian(x).T @ residual(x)

    result = minimize(
        objective, _X0, jac=grad, hess=hessian, method="trust-exact",
        options={"gtol": 1e-14, "maxiter": 5000},
    )
    x_star = result.x
    cost_star = objective(x_star)

    return Problem(
        id=f"large_residual(rho={rho})",
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        starts={"standard": _X0},
        optima=[Optimum(x_star=x_star, cost_star=cost_star)],
        kind="residuals",
        tags=frozenset(),
        probe_points=[_X0, x_star],
        source="adapted from prototype/large_residual.py",
    )
