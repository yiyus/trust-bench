import numpy as np

from trust_bench.core.problem import Optimum, Problem

M = 40
T = np.linspace(0.0, 1.0, M)
A0, K0 = 2.0, 1.3
_Y = A0 * np.exp(K0 * T)


def _residual(x):
    a, k = x
    return a * np.exp(k * T) - _Y


def _jacobian(x):
    a, k = x
    e = np.exp(k * T)
    return np.column_stack([e, a * T * e])


def _hessian(x):
    a, k = x
    e = np.exp(k * T)
    r = _residual(x)
    J = _jacobian(x)
    h = J.T @ J
    s = np.zeros((2, 2))
    s[0, 1] = s[1, 0] = np.sum(r * T * e)
    s[1, 1] = np.sum(r * a * T**2 * e)
    return h + s


PROBLEM = Problem(
    id="expdec",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([1.0, 0.5])},
    optima=[Optimum(x_star=np.array([A0, K0]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([1.0, 0.5]),
        np.array([A0, K0]),
        np.array([3.0, 2.0]),
    ],
    source="adapted from prototype/large_residual.py (rho=0)",
)
