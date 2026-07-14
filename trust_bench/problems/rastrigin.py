import numpy as np

from trust_bench.core.problem import Optimum, Problem

_A = 10.0
_N = 2


def _cost(x):
    x = np.asarray(x, dtype=float)
    return float(_A * _N + np.sum(x**2 - _A * np.cos(2 * np.pi * x)))


# Within the global minimum's basin (confirmed directly: BFGS from
# (0.3, -0.3) and closer converges to (0, 0); (0.3, -0.4) already lands
# in a neighbouring local minimum instead), unlike a start chosen to
# stress a specific failure mode - the point here is a clean, black-box
# scalar landscape for BFGS, not a hard multimodal search.
_START = np.array([0.25, -0.25])

PROBLEM = Problem(
    id="rastrigin",
    residual=_cost,
    jacobian=None,
    hessian=None,
    starts={"standard": _START},
    optima=[Optimum(x_star=np.zeros(_N), cost_star=0.0)],
    kind="scalar",
    tags=frozenset(),
    probe_points=[_START, np.zeros(_N)],
    source="Rastrigin function (a standard global-optimisation test function, not from MGH or a least-squares fit)",
)
