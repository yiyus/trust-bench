import numpy as np

from trust_bench.core.problem import Optimum, Problem


def _residual(x):
    x1, x2 = x
    return np.array([10.0 * (x2 - x1**2), 1.0 - x1])


def _jacobian(x):
    x1, x2 = x
    return np.array([[-20.0 * x1, 10.0], [-1.0, 0.0]])


def _hessian(x):
    r = _residual(x)
    J = _jacobian(x)
    h1 = np.array([[-20.0, 0.0], [0.0, 0.0]])
    return J.T @ J + r[0] * h1


PROBLEM = Problem(
    id="rosenbrock",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={
        "standard": np.array([-1.2, 1.0]),
        "far": np.array([-5.0, -5.0]),
    },
    optima=[Optimum(x_star=np.array([1.0, 1.0]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([-1.2, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([-5.0, -5.0]),
    ],
    source="Moré-Garbow-Hillstrom #1 (Rosenbrock)",
)
