import numpy as np

from trust_bench.core.problem import Optimum, Problem

_A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_X_TRUE = np.array([1.0, 2.0])
_B = _A @ _X_TRUE


def _residual(x):
    return _A @ np.asarray(x, dtype=float) - _B


def _jacobian(x):
    return _A


def _hessian(x):
    return _A.T @ _A


PROBLEM = Problem(
    id="linear",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([0.0, 0.0])},
    optima=[Optimum(x_star=_X_TRUE, cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([0.0, 0.0]),
        _X_TRUE,
        np.array([-1.0, 3.0]),
    ],
    source="baseline sanity check (this project, not from the literature)",
)
