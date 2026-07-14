import pandas as pd
import pytest

from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import capture
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.reporting.tables import results_to_dataframe, save_table
from trust_bench.studies.large_residual import RHOS, backend_results

_METRIC_COLUMNS = ["status", "message", "dist_to_opt", "cost_gap", "grad_norm_final", "n_feval"]


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
        message="`gtol` termination condition is satisfied.",
        n_iter=5,
        n_feval=10,
        n_jeval=5,
        n_heval=0,
        trace=None,
        timing=None,
        config=RunConfig(),
        provenance=capture(),
        harness_git_sha="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return RunResult(**defaults)


@pytest.fixture(scope="module")
def results():
    results, _ = backend_results()
    return results


@pytest.fixture(scope="module")
def df(results):
    return results_to_dataframe(results, key_names=["rho", "backend"])


def test_results_to_dataframe_produces_one_row_per_result(results, df):
    assert len(df) == len(results)


def test_results_to_dataframe_has_the_sweep_keys_and_metric_columns(df):
    for column in ["rho", "backend", *_METRIC_COLUMNS]:
        assert column in df.columns
    assert set(df["rho"]) == set(RHOS)


def test_save_table_writes_a_readable_csv_file(df, tmp_path):
    path = tmp_path / "table.csv"

    save_table(df, path)

    assert path.exists()
    assert path.read_text().splitlines()[0].split(",") == list(df.columns)


def test_a_declared_unsupported_rejection_is_labelled_unsupported_not_error():
    # scaling.py's own sweep catches exactly this: a backend raising
    # ValueError because it declares a requested config combination
    # unsupported (e.g. "lm does not support x_scale='jac'") - an
    # expected, passing outcome of the sweep's own probe, not a crash.
    results = {(1.0, "lm"): ValueError("lm does not support x_scale='jac'")}

    df = results_to_dataframe(results, key_names=["scale", "method"])

    assert df["status"].iloc[0] == "UNSUPPORTED"


def test_a_declared_unsupported_rejections_message_is_kept_not_discarded():
    results = {(1.0, "lm"): ValueError("lm does not support x_scale='jac'")}

    df = results_to_dataframe(results, key_names=["scale", "method"])

    assert df["message"].iloc[0] == "lm does not support x_scale='jac'"


def test_a_declared_unsupported_rejections_other_metric_fields_are_blank():
    results = {(1.0, "lm"): ValueError("lm does not support x_scale='jac'")}

    df = results_to_dataframe(results, key_names=["scale", "method"])

    for field in ["dist_to_opt", "cost_gap", "grad_norm_final", "n_feval"]:
        assert pd.isna(df[field].iloc[0])


def test_a_completed_results_message_is_its_own_termination_explanation():
    results = {(1.0, "lm"): _run_result(message="`gtol` termination condition is satisfied.")}

    df = results_to_dataframe(results, key_names=["scale", "method"])

    assert df["message"].iloc[0] == "`gtol` termination condition is satisfied."
