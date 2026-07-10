import base64
from pathlib import Path

import pandas as pd

_STYLE = """
<style>
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 2rem; background: #f6f7f6; color: #1a2420; }
  h1 { font-size: 1.5rem; margin: 0 0 1.5rem; }
  h2 { font-size: 1.05rem; margin: 2rem 0 0.5rem; border-top: 1px solid #dde3de; padding-top: 1.5rem; }
  section:first-of-type h2 { border-top: none; padding-top: 0; }
  table { border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid #dde3de; padding: 0.35rem 0.6rem; text-align: right; }
  th { background: #eef1ef; }
  th:first-child, td:first-child { text-align: left; }
  img { max-width: 100%; height: auto; border: 1px solid #dde3de; border-radius: 6px; }
  @media (prefers-color-scheme: dark) {
    body { background: #14181a; color: #e7ede9; }
    th { background: #1f2825; }
    th, td { border-color: #2b3733; }
    img { border-color: #2b3733; }
  }
</style>
"""


def build_html_report(output_dir, title="trust-bench report"):
    """Bundles every CSV table and PNG plot already written to
    output_dir into one self-contained HTML page: each table rendered
    inline, each plot embedded as a base64 image so the page is
    portable on its own.
    """
    output_dir = Path(output_dir)
    sections = []
    for path in sorted(output_dir.glob("*.csv")):
        df = pd.read_csv(path)
        sections.append(f"<section><h2>{path.stem}</h2>{df.to_html(index=False, na_rep='')}</section>")
    for path in sorted(output_dir.glob("*.png")):
        encoded = base64.b64encode(path.read_bytes()).decode()
        sections.append(
            f'<section><h2>{path.stem}</h2>'
            f'<img src="data:image/png;base64,{encoded}" alt="{path.stem}"></section>'
        )

    body = "\n".join(sections)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        f"{_STYLE}</head><body><h1>{title}</h1>{body}</body></html>"
    )


def save_html_report(html, path):
    Path(path).write_text(html)
