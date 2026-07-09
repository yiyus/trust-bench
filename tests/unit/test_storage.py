import json

from trust_bench.core.provenance import capture
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.core.storage import append, load


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
        trace=None,
        timing=None,
        config={"ftol": 1e-8},
        provenance=capture(),
        harness_git_sha="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return defaults


def test_appending_never_overwrites_an_existing_record(tmp_path):
    path = tmp_path / "results.jsonl"
    first = RunResult(**_run_result_kwargs(problem_id="rosenbrock"))
    second = RunResult(**_run_result_kwargs(problem_id="beale"))

    append(first, path)
    append(second, path)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["problem_id"] == "rosenbrock"
    assert json.loads(lines[1])["problem_id"] == "beale"


def test_load_reconstructs_a_dataframe_matching_the_stored_records(tmp_path):
    path = tmp_path / "results.jsonl"
    results = [
        RunResult(**_run_result_kwargs(problem_id="rosenbrock", cost_final=0.0)),
        RunResult(**_run_result_kwargs(problem_id="beale", cost_final=1.5)),
    ]
    for result in results:
        append(result, path)

    df = load(path)

    assert len(df) == 2
    assert list(df["problem_id"]) == ["rosenbrock", "beale"]
    assert list(df["cost_final"]) == [0.0, 1.5]
    assert df.loc[0, "status"] == "CONVERGED"
    assert df.loc[0, "provenance"]["backend_name"] == results[0].provenance.backend_name
    assert df.loc[0, "config"] == {"ftol": 1e-8}
