import numpy as np
from scipy.optimize import least_squares

from trust_bench.core.problem import Optimum, Problem

X = np.linspace(-3.0, 3.0, 30)
A0, MU0, SIGMA0 = 5.0, 1.0, 0.8
_NOISE = np.random.default_rng(20240601).normal(0.0, 0.05, len(X))
_Y = A0 * np.exp(-((X - MU0) ** 2) / (2.0 * SIGMA0**2)) + _NOISE


def _model(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _residual(x):
    a, mu, sigma = x
    return _model(X, a, mu, sigma) - _Y


def _jacobian(x):
    a, mu, sigma = x
    e = np.exp(-((X - mu) ** 2) / (2.0 * sigma**2))
    d = X - mu
    return np.column_stack([e, a * d / sigma**2 * e, a * d**2 / sigma**3 * e])


def _hessian(x):
    a, mu, sigma = x
    e = np.exp(-((X - mu) ** 2) / (2.0 * sigma**2))
    d = X - mu
    r = _residual(x)
    j = _jacobian(x)
    h = j.T @ j
    h_a_mu = d / sigma**2 * e
    h_a_sigma = d**2 / sigma**3 * e
    h_mu_mu = -a * (sigma**2 - d**2) / sigma**4 * e
    h_mu_sigma = -a * d * (2.0 * sigma**2 - d**2) / sigma**5 * e
    h_sigma_sigma = -a * d**2 * (3.0 * sigma**2 - d**2) / sigma**6 * e
    correction = np.zeros((3, 3))
    correction[0, 1] = correction[1, 0] = np.sum(r * h_a_mu)
    correction[0, 2] = correction[2, 0] = np.sum(r * h_a_sigma)
    correction[1, 1] = np.sum(r * h_mu_mu)
    correction[1, 2] = correction[2, 1] = np.sum(r * h_mu_sigma)
    correction[2, 2] = np.sum(r * h_sigma_sigma)
    return h + correction


_START = np.array([3.0, 0.0, 1.0])
_solution = least_squares(_residual, _START, jac=_jacobian, ftol=1e-14, xtol=1e-14, gtol=1e-14)
_X_STAR = _solution.x
_COST_STAR = float(_solution.cost)

PROBLEM = Problem(
    id="gaussian_peak",
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
