import numpy as np

from trust_bench.core.problem import Optimum, Problem


def _theta(x1, x2):
    base = np.arctan(x2 / x1) / (2 * np.pi)
    return base if x1 > 0 else 0.5 + base


def _residual(x):
    x1, x2, x3 = x
    theta = _theta(x1, x2)
    return np.array(
        [
            10.0 * (x3 - 10.0 * theta),
            10.0 * (np.sqrt(x1**2 + x2**2) - 1.0),
            x3,
        ]
    )


def _jacobian(x):
    x1, x2, x3 = x
    rr = x1**2 + x2**2
    dtheta_dx1 = -x2 / rr / (2 * np.pi)
    dtheta_dx2 = x1 / rr / (2 * np.pi)
    s = np.sqrt(rr)
    return np.array(
        [
            [-100.0 * dtheta_dx1, -100.0 * dtheta_dx2, 10.0],
            [10.0 * x1 / s, 10.0 * x2 / s, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _hessian(x):
    x1, x2, x3 = x
    r = _residual(x)
    J = _jacobian(x)
    h = J.T @ J
    rr = x1**2 + x2**2
    d2t_dx1x1 = 2 * x1 * x2 / rr**2 / (2 * np.pi)
    d2t_dx2x2 = -2 * x1 * x2 / rr**2 / (2 * np.pi)
    d2t_dx1x2 = (x2**2 - x1**2) / rr**2 / (2 * np.pi)
    h1 = np.zeros((3, 3))
    h1[0, 0] = -100.0 * d2t_dx1x1
    h1[1, 1] = -100.0 * d2t_dx2x2
    h1[0, 1] = h1[1, 0] = -100.0 * d2t_dx1x2
    s = np.sqrt(rr)
    d2s_dx1x1 = x2**2 / s**3
    d2s_dx2x2 = x1**2 / s**3
    d2s_dx1x2 = -x1 * x2 / s**3
    h2 = np.zeros((3, 3))
    h2[0, 0] = 10.0 * d2s_dx1x1
    h2[1, 1] = 10.0 * d2s_dx2x2
    h2[0, 1] = h2[1, 0] = 10.0 * d2s_dx1x2
    return h + r[0] * h1 + r[1] * h2


PROBLEM = Problem(
    id="helical",
    residual=_residual,
    jacobian=_jacobian,
    hessian=_hessian,
    starts={"standard": np.array([-1.0, 0.0, 0.0])},
    optima=[Optimum(x_star=np.array([1.0, 0.0, 0.0]), cost_star=0.0)],
    kind="residuals",
    tags=frozenset(),
    probe_points=[
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.5, 0.3, 0.2]),
    ],
    source="Moré-Garbow-Hillstrom #7 (Helical valley)",
)
