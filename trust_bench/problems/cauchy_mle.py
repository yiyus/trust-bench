import numpy as np
from scipy.optimize import minimize

from trust_bench.core.problem import Optimum, Problem

_TRUE_LOC, _TRUE_SCALE = 2.0, 1.5
_N_SAMPLES = 30
# Fixed, reproducible noise (numpy's PCG64 default_rng is stable across
# versions for a given seed).
_DATA = _TRUE_LOC + _TRUE_SCALE * np.random.default_rng(20240601).standard_cauchy(_N_SAMPLES)


def _neg_log_likelihood(params):
    # log_scale, not scale, is the free parameter: an MLE search over a
    # raw scale can wander non-positive (undefined log-likelihood),
    # while exp(log_scale) is always positive by construction, keeping
    # this an unconstrained problem without introducing bounds.
    loc, log_scale = params
    scale = np.exp(log_scale)
    z = (_DATA - loc) / scale
    return float(np.sum(np.log1p(z**2)) + _N_SAMPLES * (np.log(np.pi) + log_scale))


_START = np.array([0.0, 0.0])
# Cross-checked with BFGS from the same start (agrees to 6+ significant
# figures): a Cauchy MLE's negative log-likelihood has no natural
# sum-of-squares decomposition (unlike a Gaussian MLE, which reduces
# exactly to least squares), so Nelder-Mead establishes ground truth
# without assuming any gradient-based structure at all.
_solution = minimize(
    _neg_log_likelihood, _START, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=10000)
)
_X_STAR = _solution.x
_COST_STAR = float(_solution.fun)

PROBLEM = Problem(
    id="cauchy_mle",
    residual=_neg_log_likelihood,
    jacobian=None,
    hessian=None,
    starts={"standard": _START},
    optima=[Optimum(x_star=_X_STAR, cost_star=_COST_STAR)],
    kind="scalar",
    tags=frozenset(),
    probe_points=[_START, _X_STAR],
    source=(
        "this project's own scalar-cost batch: a Cauchy location-scale MLE, "
        "whose log-likelihood has no natural residual decomposition"
    ),
)
