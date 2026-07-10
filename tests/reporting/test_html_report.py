import pandas as pd

from trust_bench.reporting.html_report import build_html_report, save_html_report
from trust_bench.reporting.plots import plot_metric_vs_sweep, save_figure
from trust_bench.reporting.tables import save_table


def test_build_html_report_embeds_every_table_in_the_directory(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    save_table(df, tmp_path / "example.csv")

    html = build_html_report(tmp_path)

    assert "example" in html
    assert "<table" in html
    assert "1" in html and "3" in html


def test_build_html_report_embeds_every_plot_as_a_self_contained_image(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    fig = plot_metric_vs_sweep(df, x="x", y="y")
    save_figure(fig, tmp_path / "example.png")

    html = build_html_report(tmp_path)

    assert "example" in html
    assert "data:image/png;base64," in html


def test_save_html_report_writes_a_readable_file(tmp_path):
    html = "<html><body>hello</body></html>"
    path = tmp_path / "report.html"

    save_html_report(html, path)

    assert path.read_text() == html
