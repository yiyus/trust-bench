from trust_bench.reporting.tables import results_to_dataframe, save_table
from trust_bench.studies.large_residual import RHOS, backend_results

_METRIC_COLUMNS = ["status", "dist_to_opt", "cost_gap", "grad_norm_final", "n_feval"]


def test_results_to_dataframe_produces_one_row_per_result():
    results, _ = backend_results()

    df = results_to_dataframe(results, key_names=["rho", "backend"])

    assert len(df) == len(results)


def test_results_to_dataframe_has_the_sweep_keys_and_metric_columns():
    results, _ = backend_results()

    df = results_to_dataframe(results, key_names=["rho", "backend"])

    for column in ["rho", "backend", *_METRIC_COLUMNS]:
        assert column in df.columns
    assert set(df["rho"]) == set(RHOS)


def test_save_table_writes_a_readable_csv_file(tmp_path):
    results, _ = backend_results()
    df = results_to_dataframe(results, key_names=["rho", "backend"])
    path = tmp_path / "table.csv"

    save_table(df, path)

    assert path.exists()
    assert path.read_text().splitlines()[0].split(",") == list(df.columns)
