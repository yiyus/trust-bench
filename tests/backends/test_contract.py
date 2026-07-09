import pytest
from all_backends import BACKENDS

from trust_bench.core.result import RunResult, RunStatus
from trust_bench.problems import quadratic

PROBLEM = quadratic.PROBLEM
START = "standard"


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_solve_returns_a_well_formed_run_result(backend):
    method = next(iter(backend.capabilities().methods))

    result = backend.solve(PROBLEM, method, START, {"max_iter": 45})

    assert isinstance(result, RunResult)
    assert isinstance(result.status, RunStatus)
    assert len(result.x_final) == len(PROBLEM.starts[START])
    assert result.provenance is not None
    assert result.n_feval >= 0
    assert result.n_jeval >= 0
    assert result.n_heval >= 0


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_solve_respects_max_iter_and_returns_max_iter_status_instead_of_raising(backend):
    method = next(iter(backend.capabilities().methods))

    result = backend.solve(PROBLEM, method, START, {"max_iter": 1})

    assert result.status is RunStatus.MAX_ITER


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_eval_counts_are_non_negative_and_monotone_across_increasing_max_iter(backend):
    method = next(iter(backend.capabilities().methods))

    feval_counts = []
    jeval_counts = []
    for max_iter in [5, 15, 30]:
        result = backend.solve(PROBLEM, method, START, {"max_iter": max_iter})
        assert result.n_feval >= 0
        assert result.n_jeval >= 0
        assert result.n_heval >= 0
        feval_counts.append(result.n_feval)
        jeval_counts.append(result.n_jeval)

    assert all(a <= b for a, b in zip(feval_counts, feval_counts[1:]))
    assert all(a <= b for a, b in zip(jeval_counts, jeval_counts[1:]))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_capabilities_methods_are_consistent_with_what_solve_accepts(backend):
    for method in backend.capabilities().methods:
        backend.solve(PROBLEM, method, START, {"max_iter": 1})

    with pytest.raises(ValueError):
        backend.solve(PROBLEM, "not-a-real-method", START, {"max_iter": 1})
