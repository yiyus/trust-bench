from trust_bench.backends import BACKENDS
from trust_bench.core.result import RunStatus
from trust_bench.problems import TYPICAL_PROBLEMS
from trust_bench.studies.typical import sweep


def _method_count(backends):
    return sum(len(backend.capabilities().methods) for backend in backends)


def test_sweep_covers_every_problem_and_every_method_each_backend_declares():
    results = sweep()

    assert len(results) == len(TYPICAL_PROBLEMS) * _method_count(BACKENDS)
    for (problem_id, method, backend_name), result in results.items():
        assert result.status is not None, f"{problem_id}/{method} on {backend_name}"
        assert result.dist_to_opt is not None, f"{problem_id}/{method} on {backend_name}"


def test_every_method_converges_precisely_from_the_ordinary_start():
    # These problems are chosen to be typical, not adversarial: every
    # method a backend declares should converge cleanly here, unlike the
    # difficulty-family studies where a method failing is often the
    # point.
    results = sweep()

    for (problem_id, method, backend_name), result in results.items():
        assert result.status is RunStatus.CONVERGED, f"{problem_id}/{method} on {backend_name}"
        assert result.dist_to_opt < 1e-3, f"{problem_id}/{method} on {backend_name}"
