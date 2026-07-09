import numpy as np

from trust_bench.core.problem import Optimum, Problem

_A, _B = 3.0, -2.0
_X_STAR = np.array([_A, _B])


def make(s):
    """A consistent linear system r(x) = [x1-a, s*(x2-b)]: the Hessian's
    diagonal ratio is exactly s**2, controlling the natural scale
    disparity between the two variables.
    """
    if s <= 0.0:
        raise ValueError(f"s must be positive, got {s}")

    def residual(x):
        x1, x2 = x
        return np.array([x1 - _A, s * (x2 - _B)])

    def jacobian(x):
        return np.array([[1.0, 0.0], [0.0, s]])

    def hessian(x):
        j = jacobian(x)
        return j.T @ j

    return Problem(
        id=f"scaling(s={s})",
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        starts={"standard": np.zeros(2)},
        optima=[Optimum(x_star=_X_STAR, cost_star=0.0)],
        kind="residuals",
        tags=frozenset(),
        probe_points=[_X_STAR, _X_STAR + np.array([1.0, 1.0])],
        source="this project's own difficulty family (not from the literature)",
    )
