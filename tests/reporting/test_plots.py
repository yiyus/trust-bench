import matplotlib
import pandas as pd
import pytest

from trust_bench.reporting.capability_matrix import derive_matrix
from trust_bench.reporting.plots import plot_capability_matrix, plot_metric_vs_sweep, save_figure
from trust_bench.reporting.tables import results_to_dataframe
from trust_bench.studies.large_residual import backend_results


@pytest.fixture(scope="module")
def df():
    results, _ = backend_results()
    return results_to_dataframe(results, key_names=["rho", "backend"])


@pytest.fixture
def multi_group_df():
    return pd.DataFrame(
        [
            dict(x=1, y=10, method="a", backend="p", status="CONVERGED"),
            dict(x=2, y=20, method="a", backend="p", status="CONVERGED"),
            dict(x=3, y=30, method="a", backend="p", status="MAX_ITER"),
            dict(x=1, y=15, method="b", backend="q", status="CONVERGED"),
            dict(x=2, y=25, method="b", backend="q", status="CONVERGED"),
        ]
    )


def test_plotting_backend_is_headless():
    # The acceptance criterion's "no display" requirement: matplotlib
    # must use a backend that never needs one, regardless of the
    # environment this runs in (no DISPLAY, no GUI toolkit installed).
    assert matplotlib.get_backend().lower() == "agg"


def test_plot_metric_vs_sweep_draws_one_line_per_group(df):
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend")

    assert len(fig.axes[0].lines) == df["backend"].nunique()


def test_plot_metric_vs_sweep_draws_a_single_line_without_a_group(df):
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final")

    assert len(fig.axes[0].lines) == 1


def test_plot_metric_vs_sweep_applies_a_log_scale_on_each_axis_independently(df):
    logx_only = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", logx=True)
    assert logx_only.axes[0].get_xscale() == "log"
    assert logx_only.axes[0].get_yscale() == "linear"

    logy_only = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", logy=True)
    assert logy_only.axes[0].get_xscale() == "linear"
    assert logy_only.axes[0].get_yscale() == "log"


def test_plot_metric_vs_sweep_draws_one_line_per_group_combination(multi_group_df):
    fig = plot_metric_vs_sweep(multi_group_df, x="x", y="y", group=["method", "backend"])

    assert len(fig.axes[0].get_legend().get_texts()) == 2
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert labels == {"a/p", "b/q"}


def test_plot_metric_vs_sweep_marks_non_converged_points_distinctly(multi_group_df):
    fig = plot_metric_vs_sweep(multi_group_df, x="x", y="y", group="method", status_col="status")

    # 2 groups, one main line each, plus one hollow-marker overlay line
    # for group "a"'s single MAX_ITER point; group "b" has none.
    assert len(fig.axes[0].lines) == 3
    overlay = [line for line in fig.axes[0].lines if line.get_linestyle() == "None"]
    assert len(overlay) == 1
    assert overlay[0].get_markerfacecolor() == "none"


def test_plot_metric_vs_sweep_draws_no_overlay_when_status_col_is_not_given(multi_group_df):
    fig = plot_metric_vs_sweep(multi_group_df, x="x", y="y", group="method")

    assert len(fig.axes[0].lines) == 2


def test_plot_capability_matrix_draws_a_cell_per_field_and_backend_method_pair():
    matrix_df = derive_matrix()

    fig = plot_capability_matrix(matrix_df)

    image = fig.axes[0].images[0]
    n_fields = matrix_df["field"].nunique()
    n_columns = len(set(zip(matrix_df["backend"], matrix_df["method"])))
    assert image.get_array().shape == (n_fields, n_columns)


def test_save_figure_writes_a_non_empty_image_file(df, tmp_path):
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend", logx=True)
    path = tmp_path / "plot.png"

    save_figure(fig, path)

    assert path.exists()
    assert path.stat().st_size > 0
