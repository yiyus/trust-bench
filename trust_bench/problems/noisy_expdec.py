import numpy as np
from scipy.optimize import least_squares

from trust_bench.core.problem import Optimum, Problem

M = 40
T = np.linspace(0.0, 1.0, M)
A0, K0 = 2.0, 1.3
# Fixed, reproducible noise (numpy's PCG64 default_rng is stable across
# versions for a given seed): unlike expdec, the residual at the optimum
# is not exactly zero, so the true Hessian isn't exactly positive-
# definite-by-construction there.
_NOISE = np.random.default_rng(20240601).normal(0.0, 0.03, M)
_Y = A0 * np.exp(K0 * T) + _NOISE


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
    j = _jacobian(x)
    h = j.T @ j
    s = np.zeros((2, 2))
    s[0, 1] = s[1, 0] = np.sum(r * T * e)
    s[1, 1] = np.sum(r * a * T**2 * e)
    return h + s


_START = np.array([1.0, 0.5])
_solution = least_squares(_residual, _START, jac=_jacobian)
_X_STAR = _solution.x
_COST_STAR = float(_solution.cost)

PROBLEM = Problem(
    id="noisy_expdec",
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
