import numpy as np

from trust_bench.core.problem import Optimum, Problem

_J = np.array([[1.0, 0.0], [0.0, 2.0]])


def _residual(x):
    x1, x2 = x
    return np.array([x1, 2.0 * x2])


def _jacobian(x):
    return _J


def _hessian(x):
    return _J.T @ _J


PROBLEM = Problem(
    id="quadratic",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([1.0, -1.0])},
    optima=[Optimum(x_star=np.array([0.0, 0.0]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0]),
        np.array([3.0, 2.0]),
    ],
    source="baseline sanity check (this project, not from the literature)",
)
