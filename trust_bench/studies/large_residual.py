import numpy as np

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.metrics import basin_rate
from trust_bench.core.runner import run
from trust_bench.problems.families.large_residual import make

RHOS = [0.3, 1, 3, 10, 20, 30, 50, 70, 100, 140]

_METHOD = "lm"
_CONFIG = RunConfig(max_iter=200)
_BASIN_TOL = 1e-3


def gn_spectral_radius(problem):
    """Predicted asymptotic linear rate of undamped Gauss-Newton at the
    optimum: the spectral radius of (J^T J)^{-1} S, where S is the
    second-order term (full Hessian - J^T J) Gauss-Newton drops.

    < 1: Gauss-Newton converges linearly at this rate.
    > 1: Gauss-Newton diverges; full Newton still converges quadratically
    since it keeps S.
    """
    x_star = problem.optima[0].x_star
    j = np.asarray(problem.jacobian(x_star), dtype=float)
    jtj = j.T @ j
    s = np.asarray(problem.hessian(x_star), dtype=float) - jtj
    eig = np.linalg.eigvals(np.linalg.solve(jtj, s))
    return float(np.max(np.abs(eig)))


def undamped_gauss_newton_errors(problem, x0, max_iter=60, cap=1e6):
    """Pure (undamped) Gauss-Newton iteration x_{k+1} = x_k - (J^T J)^{-1}
    J^T r. Returns the ||x_k - x*|| error trace, starting from x0 and
    stopping early once it is clearly converged, diverged, or the normal
    equations become unsolvable.

    x0 must be close to the optimum for the spectral-radius prediction to
    apply: it is a local asymptotic rate, and this iteration has no
    global convergence guarantee.
    """
    x_star = problem.optima[0].x_star
    x = np.array(x0, dtype=float)
    errors = [float(np.linalg.norm(x - x_star))]
    for _ in range(max_iter):
        r = np.asarray(problem.residual(x), dtype=float)
        j = np.asarray(problem.jacobian(x), dtype=float)
        try:
            x = x - np.linalg.solve(j.T @ j, j.T @ r)
        except np.linalg.LinAlgError:
            break
        e = float(np.linalg.norm(x - x_star))
        errors.append(e)
        if e < 1e-14 or e > cap or not np.isfinite(e):
            break
    return errors


def backend_results(rhos=RHOS, backends=BACKENDS):
    """grad_norm_final and the basin-of-attraction rate per rho and
    backend, from the registered "standard" start, regardless of whether
    the backend exposes a trace (Section 9 item 2).
    """
    results = {}
    rates = {}
    for rho in rhos:
        problem = make(rho)
        for backend in backends:
            results[(rho, backend.name)] = run(problem, backend, _METHOD, "standard", _CONFIG)
            distances = [
                run(problem, backend, _METHOD, start, _CONFIG).dist_to_opt
                for start in problem.starts
            ]
            rates[(rho, backend.name)] = basin_rate(distances, _BASIN_TOL)
    return results, rates
