import numpy as np
import pandas as pd
import pytest

from trust_bench.reporting import colors
from trust_bench.reporting.html_report import TIER_2_INTROS, TITLES, build_html_report, save_html_report
from trust_bench.reporting.plots import plot_metric_vs_sweep, save_figure
from trust_bench.reporting.tables import save_table

_TIMING_COLUMNS = ["timing_median", "timing_mad", "timing_n_reps", "timing_warmup", "timing_thread_count"]


def _study_df(with_timing=False):
    row = dict(
        backend="scipy",
        status="CONVERGED",
        message=None,
        dist_to_opt=1e-9,
        cost_gap=1e-12,
        grad_norm_final=1e-8,
        n_feval=12,
    )
    other = dict(row, backend="trust-apl", status="FAILED")
    df = pd.DataFrame([row, other])
    for column in _TIMING_COLUMNS:
        df[column] = np.nan
    if with_timing:
        df.loc[0, ["timing_median", "timing_mad", "timing_n_reps", "timing_warmup", "timing_thread_count"]] = [
            0.0123,
            0.0004,
            5,
            1,
            1,
        ]
    return df


def _write_study(output_dir, name, with_plot=False, with_timing=False):
    df = _study_df(with_timing=with_timing)
    save_table(df, output_dir / f"{name}.csv")
    if with_plot:
        fig = plot_metric_vs_sweep(df.assign(x=[1, 2]), x="x", y="dist_to_opt", group="backend")
        save_figure(fig, output_dir / f"{name}.png")


@pytest.fixture
def populated_report(tmp_path):
    _write_study(tmp_path, "parity_scatter", with_plot=True)
    _write_study(tmp_path, "capability_frontier", with_plot=True)
    _write_study(tmp_path, "capability_matrix", with_plot=True)
    _write_study(tmp_path, "baseline")
    _write_study(tmp_path, "large_residual", with_plot=True, with_timing=True)
    _write_study(tmp_path, "robust_loss", with_plot=True)
    # bounded: table-only, no plot, matching cli.py's own real behaviour.
    _write_study(tmp_path, "bounded")
    return tmp_path


def _heading(name):
    return f"<h3>{TITLES[name]}</h3>"


def test_headline_artefacts_are_rendered_before_any_themed_study(populated_report):
    html = build_html_report(populated_report)

    headline_position = html.index(_heading("parity_scatter"))
    themed_position = html.index(_heading("baseline"))
    assert headline_position < themed_position


def test_headline_artefacts_show_only_their_plot_not_a_raw_table(populated_report):
    html = build_html_report(populated_report)

    headline_section = html[html.index(_heading("parity_scatter")) : html.index(_heading("baseline"))]
    assert "<table" not in headline_section
    assert "<img" in headline_section


def test_tier_2_studies_are_grouped_under_their_theme_headings_with_an_intro(populated_report):
    html = build_html_report(populated_report)

    assert "Common problems" in html
    assert "Difficulty axes" in html
    assert "Special capabilities" in html
    assert TIER_2_INTROS["Common problems"] in html
    assert TIER_2_INTROS["Difficulty axes"] in html
    common_problems = html.index("Common problems")
    baseline = html.index(_heading("baseline"))
    difficulty_axes = html.index("Difficulty axes")
    large_residual = html.index(_heading("large_residual"))
    special_capabilities = html.index("Special capabilities")
    robust_loss = html.index(_heading("robust_loss"))
    assert common_problems < baseline < difficulty_axes
    assert difficulty_axes < large_residual < special_capabilities
    assert special_capabilities < robust_loss


def test_every_rendered_study_has_a_one_line_caption(populated_report):
    html = build_html_report(populated_report)

    assert "floor every backend must clear" in html  # baseline
    assert "Gauss-Newton" in html  # large_residual


def test_a_study_with_no_artefacts_in_the_output_directory_is_silently_omitted(populated_report):
    html = build_html_report(populated_report)

    # dimensionality was never written to this report's output_dir (e.g.
    # skipped for a coverage gap or --skip-slow); it must not appear, and
    # building the report must not crash over its absence.
    assert "dimensionality" not in html


def test_status_values_are_rendered_as_coloured_pills_not_bare_text(populated_report):
    html = build_html_report(populated_report)

    assert colors.status_color("CONVERGED") in html
    assert colors.status_color("FAILED") in html
    assert 'class="pill"' in html or "pill" in html


def test_a_fixed_backend_colour_legend_is_always_shown(populated_report):
    html = build_html_report(populated_report)

    for name, color in colors.BACKEND_COLORS.items():
        assert name in html
        assert color in html


def test_backend_values_in_tables_are_rendered_with_their_fixed_colour(populated_report):
    html = build_html_report(populated_report)

    assert 'class="backend-cell"' in html
    assert f'style="color:{colors.backend_color("scipy")}"' in html
    assert f'style="color:{colors.backend_color("trust-apl")}"' in html


def test_timing_columns_are_displayed_as_median_pm_mad_with_units(populated_report):
    html = build_html_report(populated_report)

    assert "12.3" in html  # 0.0123s -> 12.3ms
    assert "±" in html
    assert "ms" in html


def test_a_timing_caveat_appears_when_any_study_has_timing_data(populated_report):
    html = build_html_report(populated_report)

    assert "single" in html.lower() or "not a claim" in html.lower() or "cross-language" in html.lower()


def test_no_timing_caveat_is_shown_when_no_study_measured_timing(tmp_path):
    _write_study(tmp_path, "baseline")

    html = build_html_report(tmp_path)

    assert "cross-language" not in html.lower()


def test_save_html_report_writes_a_readable_file(tmp_path):
    html = "<html><body>hello</body></html>"
    path = tmp_path / "report.html"

    save_html_report(html, path)

    assert path.read_text() == html
