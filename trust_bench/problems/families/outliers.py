import numpy as np

from trust_bench.core.problem import Optimum, Problem

_N = 20
_T = np.linspace(0.0, 1.0, _N)
_M0, _C0 = 2.0, 1.0
_CORRUPTION = 5.0
_DESIGN = np.column_stack([_T, np.ones(_N)])

TRUE_PARAMETERS = np.array([_M0, _C0])


def make(fraction):
    """Ordinary linear regression (m*t+c fit to _N points) where the
    first round(fraction*_N) points are corrupted by a fixed offset,
    nested so a larger fraction is always a superset of a smaller one's
    corruption. fraction >= 0.5 is rejected: past a majority-corrupted
    dataset, the least-squares fit starts explaining the corrupted
    majority as the new trend rather than being dragged away from the
    true parameters, so "outlier fraction" no longer describes what the
    family's own defining property assumes.
    """
    if fraction < 0.0 or fraction >= 0.5:
        raise ValueError(f"fraction must be in [0, 0.5) (a minority), got {fraction}")

    n_corrupt = round(fraction * _N)
    y = _M0 * _T + _C0
    y = y.copy()
    y[:n_corrupt] += _CORRUPTION

    def residual(x):
        m, c = x
        return (m * _T + c) - y

    def jacobian(x):
        return _DESIGN

    def hessian(x):
        return _DESIGN.T @ _DESIGN

    x_star, *_ = np.linalg.lstsq(_DESIGN, y, rcond=None)
    r_star = residual(x_star)
    cost_star = 0.5 * float(r_star @ r_star)

    return Problem(
        id=f"outliers(fraction={fraction})",
        residual=residual,
        jacobian=jacobian,
        hessian=hessian,
        starts={"standard": np.zeros(2)},
        optima=[Optimum(x_star=x_star, cost_star=cost_star)],
        kind="residuals",
        tags=frozenset(),
        probe_points=[x_star, np.zeros(2)],
        source="this project's own difficulty family (not from the literature)",
    )
