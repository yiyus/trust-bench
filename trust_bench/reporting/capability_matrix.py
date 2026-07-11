import pandas as pd

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.problems import quadratic

FIELDS = ["bounds", "analytic_hessian"]

_LEAST_SQUARES_METHODS = frozenset({"lm", "trf", "dogbox"})
_LEAST_SQUARES_BOUNDS_PROBE = ([0.5, -10.0], [10.0, 10.0])
_MINIMIZE_BOUNDS_PROBE = [(0.5, 10.0), (-10.0, 10.0)]


def _measure_bounds(backend, method):
    """True if a box constraint is genuinely respected, False if it is
    rejected or ignored, probed directly against the trivial quadratic
    problem.
    """
    # scipy is the one backend with a genuine per-method split (it
    # wraps two different underlying scipy call paths, least_squares
    # and minimize); every other backend uses the least-squares-style
    # probe universally, regardless of whether the method's name also
    # happens to exist in scipy's minimize family.
    if backend.name == "scipy" and method not in _LEAST_SQUARES_METHODS:
        bounds = _MINIMIZE_BOUNDS_PROBE
    else:
        bounds = _LEAST_SQUARES_BOUNDS_PROBE
    try:
        result = backend.solve(
            quadratic.PROBLEM, method, "standard", RunConfig(max_iter=100, bounds=bounds)
        )
    except ValueError:
        return False
    return result.status is RunStatus.CONVERGED and result.x_final[0] >= 0.5 - 1e-6


def _measure_analytic_hessian(backend, method):
    """True if solving with the problem's real analytic Hessian causes
    the backend to actually call it at least once.
    """
    result = backend.solve(quadratic.PROBLEM, method, "standard", RunConfig(max_iter=100))
    return (result.n_heval or 0) > 0


_PROBES = {
    "bounds": _measure_bounds,
    "analytic_hessian": _measure_analytic_hessian,
}


def derive_matrix(backends=BACKENDS, fields=FIELDS):
    """Cross-references each backend method's declared Capabilities
    against a live probe, per field. One row per (backend, method,
    field): declared, measured, and whether they agree.
    """
    rows = []
    for backend in backends:
        for method, caps in backend.capabilities().methods.items():
            for field in fields:
                declared = getattr(caps, field)
                measured = _PROBES[field](backend, method)
                rows.append(
                    dict(
                        backend=backend.name,
                        method=method,
                        field=field,
                        declared=declared,
                        measured=measured,
                        agrees=declared == measured,
                    )
                )
    return pd.DataFrame(rows)
