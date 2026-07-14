from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families import scaling

SCALES = [1.0, 10.0, 1e2, 1e4, 1e6, 1e8]
METHODS = ["lm", "trf", "dogbox"]
# "fixed" is a per-scale (1.0, 1.0/scale) reparameterisation matching
# scaling.make's own known anisotropy (its docstring: the Hessian's
# diagonal ratio is exactly scale**2) - the actual RunConfig.x_scale
# value passed varies with scale (see _x_scale_for), but "fixed" is
# used as the dict key/group label so the report can plot one
# continuous line across the sweep, the same way "jac" already does.
X_SCALES = [None, "jac", "fixed"]


def _x_scale_for(x_scale, scale):
    return (1.0, 1.0 / scale) if x_scale == "fixed" else x_scale


def sweep(scales=SCALES, methods=METHODS, x_scales=X_SCALES, backends=BACKENDS):
    """RunResult, or the raised ValueError, per (scale, method, x_scale,
    backend_name): the parameter set scaling.make(scale) solved with and
    without a native x_scale (SciPy's own "jac", or a fixed reparameterisation
    both backends support). Skips a (method, backend) pair the backend
    does not support; a method the backend does support but rejects a
    given x_scale for (there is no declarative capability for this,
    unlike bounds or derivative_mode) is caught and recorded instead of
    raising, matching bounded.py's own pattern.
    """
    results = {}
    for scale in scales:
        problem = scaling.make(scale)
        for backend in backends:
            supported = backend.capabilities().methods
            for method in methods:
                if method not in supported:
                    continue
                for x_scale in x_scales:
                    config = RunConfig(max_iter=200, x_scale=_x_scale_for(x_scale, scale))
                    try:
                        results[(scale, method, x_scale, backend.name)] = run(
                            problem, backend, method, "standard", config
                        )
                    except ValueError as error:
                        results[(scale, method, x_scale, backend.name)] = error

            # trust-apl's BFGS also gains a fixed pscale, but scipy's
            # own BFGS (minimize-family) has no x_scale concept at all
            # - scipy_backend.py's _solve_minimize never reads
            # config.x_scale, so including "BFGS" in the loop above
            # would silently produce a meaningless "x_scale requested
            # but never applied" row for scipy. Exercised here instead,
            # backends with a BFGS+x_scale capability only - a
            # trust-only capability worth recording on its own, no
            # scipy point of comparison needed (matching robust_loss.py's
            # own precedent for a trust-only capability).
            if "fixed" in x_scales and backend.name != "scipy" and "BFGS" in supported:
                config = RunConfig(max_iter=200, x_scale=_x_scale_for("fixed", scale))
                try:
                    results[(scale, "BFGS", "fixed", backend.name)] = run(
                        problem, backend, "BFGS", "standard", config
                    )
                except ValueError as error:
                    results[(scale, "BFGS", "fixed", backend.name)] = error
    return results
