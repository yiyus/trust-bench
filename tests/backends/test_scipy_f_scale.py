import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.core.config import RunConfig
from trust_bench.problems.families import outliers

BACKEND = SciPyBackend()
START = "standard"


def test_f_scale_is_forwarded_to_least_squares():
    # Pinned against a direct scipy call rather than a precision claim:
    # this only checks the value reaches least_squares, not that a
    # particular f_scale is the "right" one for any given loss.
    problem = outliers.make(0.2)
    config = RunConfig(max_iter=200, loss="huber", f_scale=1.345)

    result = BACKEND.solve(problem, "trf", START, config)

    x0 = np.array(problem.starts[START], dtype=float)
    expected = scipy_least_squares(
        problem.residual, x0, jac=problem.jacobian, method="trf", loss="huber", f_scale=1.345, max_nfev=200
    )

    assert result.x_final == expected.x.tolist()


def test_f_scale_left_none_keeps_scipys_own_default():
    # No f_scale set: least_squares falls back to its own default (1.0),
    # unaffected by this RunConfig field existing at all.
    problem = outliers.make(0.2)
    config = RunConfig(max_iter=200, loss="huber")

    result = BACKEND.solve(problem, "trf", START, config)

    x0 = np.array(problem.starts[START], dtype=float)
    expected = scipy_least_squares(
        problem.residual, x0, jac=problem.jacobian, method="trf", loss="huber", max_nfev=200
    )

    assert result.x_final == expected.x.tolist()
