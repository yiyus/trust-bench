from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.runner import run
from trust_bench.problems.families import dimensionality

N_VALUES = [2, 10, 100, 1000]
DENSE_METHODS = ["trust-exact", "BFGS"]
MATRIX_FREE_METHODS = ["Newton-CG", "trust-krylov", "L-BFGS-B"]
METHODS = DENSE_METHODS + MATRIX_FREE_METHODS


def sweep(n_values=N_VALUES, methods=METHODS, max_iter=200, backends=BACKENDS):
    """RunResult per (n, method, backend_name): the generalised
    Rosenbrock at each dimension, solved by every method.
    """
    results = {}
    for n in n_values:
        problem = dimensionality.make(n)
        config = RunConfig(max_iter=max_iter)
        for backend in backends:
            for method in methods:
                results[(n, method, backend.name)] = run(problem, backend, method, "standard", config)
    return results
