import dataclasses

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families import outliers

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
SCIPY_LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
# trust's own redescending losses scipy has no equivalent for. Only
# "welsch" is swept by trust_loss_precision by default: from a warm
# start, it recovers the true parameters up to 45% contamination and
# only breaks down near the 49% theoretical limit; "tukey" remains
# erratic even warm-started (its harder cutoff), and "fair" is not
# itself redescending (trust's own Loss.apln says so directly).
TRUST_LOSSES = ["welsch"]

# One explicit method per backend, not a capability-driven search:
# scipy declares more than one method that would technically satisfy
# every swept loss ("trf" and "dogbox" both do), so picking a method
# per backend explicitly is more legible than resolving that ambiguity
# implicitly.
_METHOD_FOR_BACKEND = {"scipy": "trf", "trust-apl": "lm"}
_TUKEY_C = 4.685


def _warm_start(problem):
    """An arctan-loss scipy fit: a redescending loss has no basin-of-
    attraction guarantee from an arbitrary starting point (a flat,
    zero-gradient region wherever the residual/scale ratio exceeds its
    cutoff), so both the hand-rolled IRLS reference and trust's own
    redescending losses need a start already close to a reasonable fit.
    """
    x0 = np.zeros(len(problem.starts["standard"]))
    return scipy_least_squares(problem.residual, x0, jac=problem.jacobian, loss="arctan", max_nfev=200).x


def scipy_loss_precision(fractions=FRACTIONS, losses=SCIPY_LOSSES, backends=BACKENDS):
    """Distance from the true (uncorrupted) parameters per outlier
    fraction, loss, and backend. Skips a backend this study has no
    method mapping for.
    """
    precision = {}
    for fraction in fractions:
        problem = outliers.make(fraction)
        for backend in backends:
            method = _METHOD_FOR_BACKEND.get(backend.name)
            if method is None:
                continue
            for loss in losses:
                result = run(problem, backend, method, "standard", RunConfig(max_iter=200, loss=loss))
                x_final = np.asarray(result.x_final, dtype=float)
                precision[(fraction, loss, backend.name)] = float(
                    np.linalg.norm(x_final - outliers.TRUE_PARAMETERS)
                )
    return precision


def trust_loss_precision(fractions=FRACTIONS, losses=TRUST_LOSSES, backends=BACKENDS):
    """Distance from the true (uncorrupted) parameters per outlier
    fraction, loss, and backend, for trust's own redescending losses
    scipy has no equivalent for. Warm-started (see _warm_start) and
    skips a backend this study has no method mapping for, or that
    doesn't declare support for the requested loss.
    """
    precision = {}
    for fraction in fractions:
        problem = outliers.make(fraction)
        warm_start = _warm_start(problem)
        warm_problem = dataclasses.replace(problem, starts={"warm": warm_start})
        for backend in backends:
            method = _METHOD_FOR_BACKEND.get(backend.name)
            if method is None:
                continue
            caps = backend.capabilities().methods[method]
            for loss in losses:
                if loss not in caps.losses:
                    continue
                result = run(warm_problem, backend, method, "warm", RunConfig(max_iter=200, loss=loss))
                x_final = np.asarray(result.x_final, dtype=float)
                precision[(fraction, loss, backend.name)] = float(
                    np.linalg.norm(x_final - outliers.TRUE_PARAMETERS)
                )
    return precision


def irls_tukey(problem, max_iter=50, tol=1e-10):
    """Iteratively reweighted least squares with Tukey's biweight, a
    genuinely redescending psi-function: its weight reaches exactly zero
    beyond a cutoff, unlike a bounded-but-nonzero-influence loss.

    Warm-started from SciPy's arctan-loss fit rather than plain OLS:
    Tukey's biweight has no basin-of-attraction guarantee from an
    arbitrary starting point, and an OLS start gets fooled by this
    family's high-leverage corruption pattern from a moderate outlier
    fraction onward.
    """
    warm_start = _warm_start(problem)
    design = np.asarray(problem.jacobian(warm_start), dtype=float)
    y = design @ warm_start - np.asarray(problem.residual(warm_start), dtype=float)

    beta = warm_start
    for _ in range(max_iter):
        r = y - design @ beta
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        # A near-zero scale means the current fit already leaves no
        # residual spread to reweight against; stop rather than fall
        # back to a plain, unweighted refit that would drag beta back
        # toward the corrupted points' influence.
        if scale < 1e-12:
            break
        u = r / (_TUKEY_C * scale)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        weighted_design = design * w[:, None]
        new_beta, *_ = np.linalg.lstsq(weighted_design.T @ design, weighted_design.T @ y, rcond=None)
        if np.linalg.norm(new_beta - beta) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def irls_precision(fractions=FRACTIONS):
    """Distance from the true (uncorrupted) parameters per outlier
    fraction, using the hand-rolled Tukey-biweight IRLS reference.
    """
    return {
        fraction: float(np.linalg.norm(irls_tukey(outliers.make(fraction)) - outliers.TRUE_PARAMETERS))
        for fraction in fractions
    }
