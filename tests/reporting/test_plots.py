import matplotlib
import pandas as pd
import pytest

from trust_bench.reporting.capability_matrix import derive_matrix
from trust_bench.reporting.plots import (
    plot_capability_frontier,
    plot_capability_matrix,
    plot_metric_by_category,
    plot_metric_vs_sweep,
    plot_parity_scatter,
    save_figure,
)
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


@pytest.fixture
def category_df():
    return pd.DataFrame(
        [
            dict(problem_id="p1", y=1.0, method="a", backend="x", status="CONVERGED"),
            dict(problem_id="p1", y=2.0, method="b", backend="x", status="MAX_ITER"),
            dict(problem_id="p2", y=3.0, method="a", backend="x", status="CONVERGED"),
            dict(problem_id="p2", y=4.0, method="b", backend="x", status="CONVERGED"),
        ]
    )


@pytest.fixture
def parity_df():
    return pd.DataFrame(
        [
            dict(x=1e-10, y=1e-9, converged=True, study="baseline"),
            dict(x=1e-8, y=1e-7, converged=True, study="baseline"),
            dict(x=1e-6, y=1.0, converged=False, study="typical"),
            dict(x=1e-6, y=1e-6, converged=True, study="typical"),
        ]
    )


@pytest.fixture
def frontier_panels_fixture():
    two_backend = pd.DataFrame(
        [
            dict(x=1, y=10, backend="scipy"),
            dict(x=2, y=20, backend="scipy"),
            dict(x=1, y=15, backend="trust-apl"),
            dict(x=2, y=5, backend="trust-apl"),
        ]
    )
    one_backend = pd.DataFrame([dict(x=1, y=100, backend="scipy"), dict(x=2, y=200, backend="scipy")])
    return {"panel_a": (two_backend, "x", "y"), "panel_b": (one_backend, "x", "y")}


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


def test_plot_metric_by_category_draws_one_bar_cluster_per_category(category_df):
    fig = plot_metric_by_category(category_df, category="problem_id", y="y", group=["method", "backend"])

    # 2 categories (p1, p2), 2 groups (a/x, b/x) each: 4 bars total.
    assert len(fig.axes[0].patches) == 4
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert labels == {"a/x", "b/x"}


def test_plot_metric_by_category_hatches_non_converged_bars(category_df):
    fig = plot_metric_by_category(
        category_df, category="problem_id", y="y", group=["method", "backend"], status_col="status"
    )

    hatched = [bar for bar in fig.axes[0].patches if bar.get_hatch() is not None]
    assert len(hatched) == 1


def test_plot_metric_by_category_draws_no_hatching_when_status_col_is_not_given(category_df):
    fig = plot_metric_by_category(category_df, category="problem_id", y="y", group=["method", "backend"])

    hatched = [bar for bar in fig.axes[0].patches if bar.get_hatch() is not None]
    assert len(hatched) == 0


def test_plot_capability_matrix_draws_a_cell_per_field_and_backend_method_pair():
    matrix_df = derive_matrix()

    fig = plot_capability_matrix(matrix_df)

    image = fig.axes[0].images[0]
    n_fields = matrix_df["field"].nunique()
    n_columns = len(set(zip(matrix_df["backend"], matrix_df["method"])))
    assert image.get_array().shape == (n_fields, n_columns)


def test_plot_parity_scatter_draws_a_diagonal_reference_line(parity_df):
    fig = plot_parity_scatter(parity_df, x="x", y="y", converged_col="converged")

    reference = [line for line in fig.axes[0].lines if line.get_label() == "y = x"]
    assert len(reference) == 1
    assert reference[0].get_linestyle() == "--"


def test_plot_parity_scatter_marks_non_converged_rows_distinctly(parity_df):
    fig = plot_parity_scatter(parity_df, x="x", y="y", converged_col="converged")

    marked = [line for line in fig.axes[0].lines if line.get_marker() == "x"]
    assert len(marked) == 1
    assert len(marked[0].get_xdata()) == 1


def test_plot_parity_scatter_draws_no_marked_points_without_a_converged_col(parity_df):
    fig = plot_parity_scatter(parity_df, x="x", y="y")

    marked = [line for line in fig.axes[0].lines if line.get_marker() == "x"]
    assert len(marked) == 0


def test_plot_parity_scatter_groups_by_the_given_column(parity_df):
    fig = plot_parity_scatter(parity_df, x="x", y="y", converged_col="converged", group="study")

    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert {"baseline", "typical"} <= labels


def test_plot_parity_scatter_uses_log_scale_on_both_axes(parity_df):
    fig = plot_parity_scatter(parity_df, x="x", y="y", converged_col="converged")

    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"


def test_plot_capability_frontier_draws_one_subplot_per_panel(frontier_panels_fixture):
    fig = plot_capability_frontier(frontier_panels_fixture)

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["panel_a", "panel_b"]


def test_plot_capability_frontier_draws_one_line_per_backend_per_panel(frontier_panels_fixture):
    fig = plot_capability_frontier(frontier_panels_fixture)

    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[1].lines) == 1


def test_save_figure_writes_a_non_empty_image_file(df, tmp_path):
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend", logx=True)
    path = tmp_path / "plot.png"

    save_figure(fig, path)

    assert path.exists()
    assert path.stat().st_size > 0
