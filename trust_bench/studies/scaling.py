from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families import scaling

SCALES = [1.0, 10.0, 1e2, 1e4, 1e6, 1e8]
METHODS = ["lm", "trf", "dogbox"]
X_SCALES = [None, "jac"]


def sweep(scales=SCALES, methods=METHODS, x_scales=X_SCALES, backends=BACKENDS):
    """RunResult per (scale, method, x_scale, backend_name): the
    parameter set scaling.make(scale) solved with and without SciPy's
    x_scale.
    """
    results = {}
    for scale in scales:
        problem = scaling.make(scale)
        for backend in backends:
            for method in methods:
                for x_scale in x_scales:
                    config = RunConfig(max_iter=200, x_scale=x_scale)
                    results[(scale, method, x_scale, backend.name)] = run(
                        problem, backend, method, "standard", config
                    )
    return results
