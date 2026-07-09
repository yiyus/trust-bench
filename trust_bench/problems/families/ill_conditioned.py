import numpy as np

from trust_bench.core.problem import Optimum, Problem

_THETA = np.pi / 6
_R = np.array([[np.cos(_THETA), -np.sin(_THETA)], [np.sin(_THETA), np.cos(_THETA)]])
_X_TRUE = np.array([2.0, 3.0])


def make(kappa):
    """A consistent linear system r(x) = A@x - b with cond(A) == kappa
    exactly. A is a rotated (not axis-aligned) diagonal, so the
    ill-conditioning cannot be sidestepped by treating each variable
    independently.
    """
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1 (a condition number), got {kappa}")

    a = np.diag([kappa, 1.0]) @ _R
    b = a @ _X_TRUE

    def residual(x):
        return a @ np.asarray(x, dtype=float) - b

    def jacobian(x):
        return a

    def hessian(x):
        return a.T @ a

    return Problem(
        id=f"ill_conditioned(kappa={kappa})",
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        starts={"standard": np.zeros(2)},
        optima=[Optimum(x_star=_X_TRUE, cost_star=0.0)],
        kind="residuals",
        tags=frozenset(),
        probe_points=[_X_TRUE, _X_TRUE + 1.0],
        source="this project's own difficulty family (not from the literature)",
    )
