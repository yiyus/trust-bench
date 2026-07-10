import matplotlib

from trust_bench.reporting.plots import plot_metric_vs_sweep, save_figure
from trust_bench.reporting.tables import results_to_dataframe
from trust_bench.studies.large_residual import backend_results


def test_plotting_backend_is_headless():
    # The acceptance criterion's "no display" requirement: matplotlib
    # must use a backend that never needs one, regardless of the
    # environment this runs in (no DISPLAY, no GUI toolkit installed).
    assert matplotlib.get_backend().lower() == "agg"


def test_plot_metric_vs_sweep_draws_one_line_per_group():
    results, _ = backend_results()
    df = results_to_dataframe(results, key_names=["rho", "backend"])

    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend")

    assert len(fig.axes[0].lines) == df["backend"].nunique()


def test_save_figure_writes_a_non_empty_image_file(tmp_path):
    results, _ = backend_results()
    df = results_to_dataframe(results, key_names=["rho", "backend"])
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend", logx=True)
    path = tmp_path / "plot.png"

    save_figure(fig, path)

    assert path.exists()
    assert path.stat().st_size > 0
