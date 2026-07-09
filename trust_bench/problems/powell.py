import numpy as np

from trust_bench.core.problem import Optimum, Problem

_SQRT5 = np.sqrt(5.0)
_SQRT10 = np.sqrt(10.0)


def _residual(x):
    x1, x2, x3, x4 = x
    return np.array(
        [
            x1 + 10.0 * x2,
            _SQRT5 * (x3 - x4),
            (x2 - 2.0 * x3) ** 2,
            _SQRT10 * (x1 - x4) ** 2,
        ]
    )


def _jacobian(x):
    x1, x2, x3, x4 = x
    u = x2 - 2.0 * x3
    v = x1 - x4
    return np.array(
        [
            [1.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, _SQRT5, -_SQRT5],
            [0.0, 2.0 * u, -4.0 * u, 0.0],
            [2.0 * _SQRT10 * v, 0.0, 0.0, -2.0 * _SQRT10 * v],
        ]
    )


def _hessian(x):
    r = _residual(x)
    J = _jacobian(x)
    h = J.T @ J
    h3 = np.zeros((4, 4))
    h3[1, 1] = 2.0
    h3[1, 2] = h3[2, 1] = -4.0
    h3[2, 2] = 8.0
    h4 = np.zeros((4, 4))
    h4[0, 0] = 2.0 * _SQRT10
    h4[0, 3] = h4[3, 0] = -2.0 * _SQRT10
    h4[3, 3] = 2.0 * _SQRT10
    return h + r[2] * h3 + r[3] * h4


PROBLEM = Problem(
    id="powell",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([3.0, -1.0, 0.0, 1.0])},
    optima=[Optimum(x_star=np.array([0.0, 0.0, 0.0, 0.0]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([3.0, -1.0, 0.0, 1.0]),
        np.array([1.0, 0.5, -0.5, 0.2]),
    ],
    source="Moré-Garbow-Hillstrom #13 (Powell singular)",
)
