import numpy as np

from trust_bench.core.problem import Optimum, Problem


def make(n):
    """The generalised Rosenbrock at dimension n: pairs of residuals per
    two variables, x* = (1,...,1), cost* = 0. n must be even (the
    residual pairing requires it).
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even, got {n}")

    def residual(x):
        x = np.asarray(x, dtype=float)
        r = np.empty(n)
        r[0::2] = 10.0 * (x[1::2] - x[0::2] ** 2)
        r[1::2] = 1.0 - x[0::2]
        return r

    def jacobian(x):
        x = np.asarray(x, dtype=float)
        j = np.zeros((n, n))
        for i in range(0, n, 2):
            j[i, i] = -20.0 * x[i]
            j[i, i + 1] = 10.0
            j[i + 1, i] = -1.0
        return j

    def hessian(x):
        x = np.asarray(x, dtype=float)
        r = residual(x)
        j = jacobian(x)
        h = j.T @ j
        for i in range(0, n, 2):
            h[i, i] += r[i] * -20.0
        return h

    standard_start = np.tile([-1.2, 1.0], n // 2)

    return Problem(
        id=f"dimensionality(n={n})",
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        starts={"standard": standard_start},
        optima=[Optimum(x_star=np.ones(n), cost_star=0.0)],
        kind="residuals",
        tags=frozenset(),
        probe_points=[np.ones(n), np.full(n, 0.7)],
        source="generalised Moré-Garbow-Hillstrom #1 (Rosenbrock)",
    )
