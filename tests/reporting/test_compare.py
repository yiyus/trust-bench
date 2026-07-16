from trust_bench.core.provenance import EnvProvenance
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.core.storage import append, load
from trust_bench.reporting.compare import REGRESSION, STABLE, compare, drift_summary


def _provenance(**overrides):
    defaults = dict(
        backend_name="scipy",
        backend_version="1.11.0",
        language_runtime="CPython 3.11.0",
        blas_lapack="openblas",
        os="Linux 6.0",
        cpu_model="generic",
        cpu_count=4,
        machine_fingerprint="fp-a",
    )
    defaults.update(overrides)
    return EnvProvenance(**defaults)


def _run_result(**overrides):
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
        trace=None,
        timing=None,
        config={"ftol": 1e-8},
        provenance=_provenance(),
        harness_git_sha="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def _write(results, path):
    for result in results:
        append(result, path)
    return load(path)


def test_identical_results_classify_as_stable(tmp_path):
    baseline = _write([_run_result()], tmp_path / "baseline.jsonl")
    candidate = _write([_run_result()], tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == STABLE


def test_a_tier1_metric_change_is_classified_as_a_regression(tmp_path):
    baseline = _write([_run_result(dist_to_opt=0.0)], tmp_path / "baseline.jsonl")
    candidate = _write([_run_result(dist_to_opt=0.5)], tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == REGRESSION
    assert "dist_to_opt" in compared["changed_tier1_metrics"].iloc[0]


def test_a_status_change_is_classified_as_a_regression(tmp_path):
    baseline = _write([_run_result(status=RunStatus.CONVERGED)], tmp_path / "baseline.jsonl")
    candidate = _write([_run_result(status=RunStatus.STALLED)], tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == REGRESSION
    assert "status" in compared["changed_tier1_metrics"].iloc[0]


def test_a_tier1_metric_change_within_tolerance_is_not_a_regression(tmp_path):
    baseline = _write([_run_result(dist_to_opt=1.0)], tmp_path / "baseline.jsonl")
    candidate = _write([_run_result(dist_to_opt=1.0 + 1e-12)], tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == STABLE


def test_a_tier3_timing_change_with_no_tier1_change_is_classified_as_drift(tmp_path):
    from trust_bench.core.result import TimingStats

    baseline_timing = TimingStats(median=0.10, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    candidate_timing = TimingStats(median=0.20, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    baseline = _write([_run_result(timing=baseline_timing)], tmp_path / "baseline.jsonl")
    candidate = _write([_run_result(timing=candidate_timing)], tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == "drift"


def test_drift_rows_carry_machine_fingerprint_and_backend_version(tmp_path):
    from trust_bench.core.result import TimingStats

    baseline_timing = TimingStats(median=0.10, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    candidate_timing = TimingStats(median=0.20, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    baseline = _write(
        [_run_result(timing=baseline_timing, provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.0"))],
        tmp_path / "baseline.jsonl",
    )
    candidate = _write(
        [_run_result(timing=candidate_timing, provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.1"))],
        tmp_path / "candidate.jsonl",
    )

    compared = compare(baseline, candidate)

    assert compared["machine_fingerprint"].iloc[0] == "fp-x"
    assert compared["backend_version"].iloc[0] == "1.1"


def test_a_tier1_regression_takes_priority_over_a_concurrent_tier3_drift(tmp_path):
    from trust_bench.core.result import TimingStats

    baseline_timing = TimingStats(median=0.10, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    candidate_timing = TimingStats(median=0.20, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    baseline = _write(
        [_run_result(dist_to_opt=0.0, timing=baseline_timing)], tmp_path / "baseline.jsonl"
    )
    candidate = _write(
        [_run_result(dist_to_opt=0.5, timing=candidate_timing)], tmp_path / "candidate.jsonl"
    )

    compared = compare(baseline, candidate)

    assert compared["classification"].iloc[0] == REGRESSION


def test_drift_summary_groups_by_machine_fingerprint_and_backend_version(tmp_path):
    from trust_bench.core.result import TimingStats

    baseline_timing = TimingStats(median=0.10, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    candidate_timing = TimingStats(median=0.20, mad=0.001, n_reps=5, warmup=1, thread_count=1)
    baseline_results = [
        _run_result(
            problem_id="rosenbrock",
            timing=baseline_timing,
            provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.1"),
        ),
        _run_result(
            problem_id="beale",
            timing=baseline_timing,
            provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.1"),
        ),
    ]
    candidate_results = [
        _run_result(
            problem_id="rosenbrock",
            timing=candidate_timing,
            provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.1"),
        ),
        _run_result(
            problem_id="beale",
            timing=candidate_timing,
            provenance=_provenance(machine_fingerprint="fp-x", backend_version="1.1"),
        ),
    ]
    baseline = _write(baseline_results, tmp_path / "baseline.jsonl")
    candidate = _write(candidate_results, tmp_path / "candidate.jsonl")

    compared = compare(baseline, candidate)
    summary = drift_summary(compared)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["machine_fingerprint"] == "fp-x"
    assert row["backend_version"] == "1.1"
    assert row["n_drifted"] == 2
