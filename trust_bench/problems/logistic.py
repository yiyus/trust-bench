import numpy as np
from scipy.optimize import least_squares

from trust_bench.core.problem import Optimum, Problem

# Log-spaced doses, a classic dose-response design (9 points, two per
# decade plus the endpoints).
X = np.logspace(-2, 2, 9)
A0, B0, C0, D0 = 0.0, 1.5, 2.0, 100.0
_NOISE = np.random.default_rng(20240601).normal(0.0, 1.5, len(X))


def _model(x, a, b, c, d):
    return d + (a - d) / (1.0 + (x / c) ** b)


_Y = _model(X, A0, B0, C0, D0) + _NOISE


def _residual(x):
    a, b, c, d = x
    return _model(X, a, b, c, d) - _Y


def _jacobian(x):
    a, b, c, d = x
    ln_ratio = np.log(X / c)
    cosh_half = np.cosh(b * ln_ratio / 2.0)
    u = np.exp(b * ln_ratio)
    weight = 1.0 / (1.0 + u)
    d_a = weight
    d_d = 1.0 - weight
    d_b = -ln_ratio * (a - d) / (4.0 * cosh_half**2)
    d_c = b * (a - d) / (4.0 * c * cosh_half**2)
    return np.column_stack([d_a, d_b, d_c, d_d])


def _hessian(x):
    a, b, c, d = x
    r = _residual(x)
    j = _jacobian(x)
    h = j.T @ j
    ln_ratio = np.log(X / c)
    cosh_half = np.cosh(b * ln_ratio / 2.0)
    sinh_half = np.sinh(b * ln_ratio / 2.0)
    h_ab = -ln_ratio / (4.0 * cosh_half**2)
    h_ac = b / (4.0 * c * cosh_half**2)
    h_bb = ln_ratio**2 * (a - d) * sinh_half / (4.0 * cosh_half**3)
    h_bc = -(a - d) * (b * ln_ratio * sinh_half - cosh_half) / (4.0 * c * cosh_half**3)
    h_bd = ln_ratio / (4.0 * cosh_half**2)
    h_cc = b * (a - d) * (b * sinh_half - cosh_half) / (4.0 * c**2 * cosh_half**3)
    h_cd = -b / (4.0 * c * cosh_half**2)
    correction = np.zeros((4, 4))
    correction[0, 1] = correction[1, 0] = np.sum(r * h_ab)
    correction[0, 2] = correction[2, 0] = np.sum(r * h_ac)
    correction[1, 1] = np.sum(r * h_bb)
    correction[1, 2] = correction[2, 1] = np.sum(r * h_bc)
    correction[1, 3] = correction[3, 1] = np.sum(r * h_bd)
    correction[2, 2] = np.sum(r * h_cc)
    correction[2, 3] = correction[3, 2] = np.sum(r * h_cd)
    return h + correction


_START = np.array([10.0, 1.0, 1.0, 90.0])
_solution = least_squares(_residual, _START, jac=_jacobian, ftol=1e-14, xtol=1e-14, gtol=1e-14)
_X_STAR = _solution.x
_COST_STAR = float(_solution.cost)

PROBLEM = Problem(
    id="logistic",
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
