import numpy as np

from trust_bench.core.problem import Optimum, Problem

_Y = np.array([1.5, 2.25, 2.625])
_I = np.array([1, 2, 3])


def _residual(x):
    x1, x2 = x
    return _Y - x1 + x1 * x2**_I


def _jacobian(x):
    x1, x2 = x
    d_dx1 = -1.0 + x2**_I
    d_dx2 = x1 * _I * x2 ** (_I - 1)
    return np.column_stack([d_dx1, d_dx2])


def _hessian(x):
    x1, x2 = x
    r = _residual(x)
    J = _jacobian(x)
    h = J.T @ J
    for i, ri in zip(_I, r):
        cross = i * x2 ** (i - 1)
        dxx2 = x1 * i * (i - 1) * x2 ** (i - 2) if i >= 2 else 0.0
        h += ri * np.array([[0.0, cross], [cross, dxx2]])
    return h


PROBLEM = Problem(
    id="beale",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([1.0, 1.0])},
    optima=[Optimum(x_star=np.array([3.0, 0.5]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([1.0, 1.0]),
        np.array([3.0, 0.5]),
        np.array([2.0, 0.2]),
        np.array([0.5, -0.5]),
    ],
    source="Moré-Garbow-Hillstrom #5 (Beale)",
)
