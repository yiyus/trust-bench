import json

import pytest

from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import capture
from trust_bench.core.result import RunResult, RunStatus, TimingStats


def _run_result_kwargs(**overrides):
    defaults = dict(
        problem_id="rosenbrock",
        backend="scipy",
        method="lm",
        start="standard",
        x_final=[1.0, 1.0],
        cost_final=0.0,
        dist_to_opt=0.0,
        cost_gap=0.0,
        grad_norm_final=0.0,
        status=RunStatus.CONVERGED,
        n_iter=5,
        n_feval=10,
        n_jeval=5,
        n_heval=0,
        trace=[[-1.2, 1.0], [1.0, 1.0]],
        timing=TimingStats(median=0.01, mad=0.001, n_reps=5, warmup=2, thread_count=1),
        config=RunConfig(tolerance=1e-8),
        provenance=capture(),
        harness_git_sha="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return defaults


def test_timing_stats_round_trips_through_json_serialisation():
    stats = TimingStats(median=0.01, mad=0.001, n_reps=5, warmup=2, thread_count=1)

    restored = TimingStats.from_dict(json.loads(json.dumps(stats.to_dict())))

    assert restored == stats


def test_run_result_round_trips_through_json_serialisation():
    result = RunResult(**_run_result_kwargs())

    restored = RunResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result


def test_run_result_round_trips_with_no_timing_stats():
    result = RunResult(**_run_result_kwargs(timing=None))

    restored = RunResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result


def test_run_result_message_defaults_to_none():
    result = RunResult(**_run_result_kwargs())

    assert result.message is None


def test_run_result_round_trips_a_message():
    result = RunResult(**_run_result_kwargs(message="Unknown problem_id: not_a_family(x=1.0)"))

    restored = RunResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result
    assert restored.message == "Unknown problem_id: not_a_family(x=1.0)"


def test_constructing_a_run_result_without_a_provenance_argument_fails():
    kwargs = _run_result_kwargs()
    del kwargs["provenance"]

    with pytest.raises(TypeError):
        RunResult(**kwargs)


def test_constructing_a_run_result_with_a_none_provenance_fails():
    with pytest.raises(TypeError):
        RunResult(**_run_result_kwargs(provenance=None))
