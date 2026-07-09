from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families.ill_conditioned import make

KAPPAS = [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
# exact trust-region, CG/Krylov, dense quasi-Newton (Section 9 item 3)
METHODS = ["trust-exact", "Newton-CG", "BFGS"]
_CONFIG = RunConfig(max_iter=200)


def sweep(kappas=KAPPAS, methods=METHODS, backends=BACKENDS):
    """RunResult per (kappa, method, backend_name), recording success
    (status) and precision (dist_to_opt) across a condition-number
    sweep. Skips a (method, backend) pair the backend does not support.
    """
    results = {}
    for kappa in kappas:
        problem = make(kappa)
        for backend in backends:
            supported = backend.capabilities().methods
            for method in methods:
                if method not in supported:
                    continue
                results[(kappa, method, backend.name)] = run(
                    problem, backend, method, "standard", _CONFIG
                )
    return results
