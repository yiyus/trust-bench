from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families import scaling

SCALES = [1.0, 10.0, 1e2, 1e4, 1e6, 1e8]
METHODS = ["lm", "trf", "dogbox"]
X_SCALES = [None, "jac"]


def sweep(scales=SCALES, methods=METHODS, x_scales=X_SCALES, backends=BACKENDS):
    """RunResult, or the raised ValueError, per (scale, method, x_scale,
    backend_name): the parameter set scaling.make(scale) solved with and
    without SciPy's x_scale. Skips a (method, backend) pair the backend
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
                    config = RunConfig(max_iter=200, x_scale=x_scale)
                    try:
                        results[(scale, method, x_scale, backend.name)] = run(
                            problem, backend, method, "standard", config
                        )
                    except ValueError as error:
                        results[(scale, method, x_scale, backend.name)] = error
    return results
