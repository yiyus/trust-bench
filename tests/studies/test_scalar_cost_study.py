from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.problems import SCALAR_PROBLEMS
from trust_bench.studies.scalar_cost import METHODS, sweep


def test_sweep_covers_every_problem_method_and_backend():
    results = sweep()

    assert len(results) == len(SCALAR_PROBLEMS) * len(METHODS) * len(BACKENDS)


def test_bfgs_and_l_bfgs_b_converge_on_every_scalar_problem():
    results = sweep()

    for problem in SCALAR_PROBLEMS:
        for method in METHODS:
            result = results[(problem.id, method, BACKENDS[0].name)]
            assert result.status is RunStatus.CONVERGED, f"{problem.id}/{method}"
            assert result.dist_to_opt < 1e-3, f"{problem.id}/{method}"
