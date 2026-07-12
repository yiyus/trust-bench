import numpy as np
from scipy.optimize import least_squares

from trust_bench.core.problem import Optimum, Problem

S = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
VMAX0, KM0 = 10.0, 3.0
_NOISE = np.random.default_rng(20240601).normal(0.0, 0.1, len(S))
_Y = VMAX0 * S / (KM0 + S) + _NOISE


def _residual(x):
    vmax, km = x
    return vmax * S / (km + S) - _Y


def _jacobian(x):
    vmax, km = x
    denom = km + S
    return np.column_stack([S / denom, -vmax * S / denom**2])


def _hessian(x):
    vmax, km = x
    denom = km + S
    r = _residual(x)
    j = _jacobian(x)
    h = j.T @ j
    d2v_dkm2 = 2.0 * vmax * S / denom**3
    d2v_dvmax_dkm = -S / denom**2
    correction = np.zeros((2, 2))
    correction[0, 1] = correction[1, 0] = np.sum(r * d2v_dvmax_dkm)
    correction[1, 1] = np.sum(r * d2v_dkm2)
    return h + correction


_START = np.array([5.0, 1.0])
_solution = least_squares(_residual, _START, jac=_jacobian)
_X_STAR = _solution.x
_COST_STAR = float(_solution.cost)

PROBLEM = Problem(
    id="michaelis_menten",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": _START},
    optima=[Optimum(x_star=_X_STAR, cost_star=_COST_STAR)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[_START, _X_STAR],
    source="this project's own typical-problems batch (not from the literature)",
)
