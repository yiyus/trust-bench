import base64
from pathlib import Path

import pandas as pd

from trust_bench.reporting.colors import BACKEND_COLORS, status_color

_STYLE = """
<style>
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 2rem; background: #f6f7f6; color: #1a2420; }
  h1 { font-size: 1.5rem; margin: 0 0 1.5rem; }
  h2 { font-size: 1.2rem; margin: 2.5rem 0 1rem; border-top: 2px solid #dde3de; padding-top: 1.5rem; }
  section:first-of-type h2 { border-top: none; padding-top: 0; }
  h3 { font-size: 1rem; margin: 1.5rem 0 0.2rem; }
  .caption { font-size: 0.85rem; color: #4d5852; margin: 0 0 0.6rem; }
  table { border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid #dde3de; padding: 0.35rem 0.6rem; text-align: right; }
  th { background: #eef1ef; }
  th:first-child, td:first-child { text-align: left; }
  img { max-width: 100%; height: auto; border: 1px solid #dde3de; border-radius: 6px; margin-top: 0.6rem; }
  .pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; color: white; font-size: 0.78rem; }
  .legend { margin: 0 0 1.5rem; font-size: 0.85rem; }
  .legend-item { display: inline-flex; align-items: center; margin-right: 1.2rem; }
  .swatch { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; margin-right: 0.4rem; }
  .timing-caveat { font-size: 0.8rem; color: #4d5852; font-style: italic; margin: 0 0 1.5rem; }
  @media (prefers-color-scheme: dark) {
    body { background: #14181a; color: #e7ede9; }
    th { background: #1f2825; }
    th, td { border-color: #2b3733; }
    img { border-color: #2b3733; }
    .caption, .timing-caveat { color: #a9b3ad; }
  }
</style>
"""

# Tier 1: headline artefacts a reader should see first, before any
# per-study detail - the "at a glance" answer to "do these backends
# agree, and where does each one's frontier lie".
TIER_1 = ["parity_scatter", "capability_frontier", "capability_matrix"]

# Tier 2: every other study, grouped by what question it answers, in
# display order.
TIER_2_GROUPS = [
    ("Common problems", ["baseline", "typical", "derivative_source"]),
    ("Difficulty axes", ["ill_conditioning", "scaling", "large_residual", "dimensionality"]),
    ("Special capabilities", ["robust_loss", "bounded"]),
]

# One-line captions, adapted from docs/plans/trust-bench.md's own study
# descriptions. A study without an entry here (there shouldn't be one,
# outside of scalar_cost, which this report never renders) gets no
# caption rather than a KeyError.
CAPTIONS = {
    "baseline": (
        "Regression and parity across backends on the canonical problem set "
        "- the floor every backend must clear."
    ),
    "typical": "A representative day-to-day mix of problems, methods, and starting points.",
    "derivative_source": "Analytic vs finite-difference Jacobians: cost in evaluations and precision.",
    "ill_conditioning": (
        "Sweeping the condition number of J/Hessian separates exact trust-region, "
        "Krylov/CG, and dense quasi-Newton behaviour."
    ),
    "scaling": "Parameters spanning many orders of magnitude, with and without adaptive scaling.",
    "large_residual": "Sweeping residual size traces the predicted vs measured Gauss-Newton failure boundary.",
    "dimensionality": (
        "Generalised Rosenbrock at increasing dimension: dense-Hessian methods against matrix-free ones."
    ),
    "robust_loss": "Sweeping outlier fraction across loss functions from L2 to redescending Tukey/Welsch.",
    "bounded": "Box constraints: inactive, active-at-boundary, and infeasible starts.",
    "parity_scatter": "Do backends agree? A tight diagonal cloud says yes, with no backend framed as better.",
    "capability_frontier": "Where each backend's line departs from a flat, converged baseline and starts climbing.",
    "capability_matrix": "Declared capabilities checked against measured behaviour, backend by backend.",
}

# Auxiliary tables rendered inside their parent study's own section,
# rather than as a Tier-2 entry of their own.
AUX_TABLES = {
    "baseline": ["baseline_basin_rates"],
    "large_residual": ["large_residual_basin_rates"],
}

TIMING_CAVEAT = (
    "Timing is a single-machine, single-snapshot reading: it supports no bare "
    "cross-language claim that one backend is faster than another, only a "
    "within-machine comparison at the time this report was generated."
)

_TIMING_FIELDS = ["timing_median", "timing_mad", "timing_n_reps", "timing_warmup", "timing_thread_count"]


def _format_timing(median, mad):
    if pd.isna(median):
        return ""
    return f"{median * 1000:.1f} ± {mad * 1000:.1f} ms"


def _render_table(df):
    df = df.copy()
    has_timing = "timing_median" in df.columns and df["timing_median"].notna().any()
    if "timing_median" in df.columns:
        df["timing"] = [_format_timing(median, mad) for median, mad in zip(df["timing_median"], df["timing_mad"])]
        df = df.drop(columns=[field for field in _TIMING_FIELDS if field in df.columns])
    if "status" in df.columns:
        df["status"] = df["status"].map(
            lambda value: f'<span class="pill" style="background:{status_color(value)}">{value}</span>'
        )
    return df.to_html(index=False, na_rep="", escape=False), has_timing


def _plot_html(output_dir, name):
    png_path = output_dir / f"{name}.png"
    if not png_path.exists():
        return ""
    encoded = base64.b64encode(png_path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{name}">'


def _study_section(output_dir, name):
    csv_path = output_dir / f"{name}.csv"
    if not csv_path.exists():
        return None
    table_html, has_timing = _render_table(pd.read_csv(csv_path))
    parts = [f"<h3>{name}</h3>", f'<p class="caption">{CAPTIONS.get(name, "")}</p>', table_html]
    for aux_name in AUX_TABLES.get(name, []):
        aux_path = output_dir / f"{aux_name}.csv"
        if aux_path.exists():
            aux_html, aux_timing = _render_table(pd.read_csv(aux_path))
            has_timing = has_timing or aux_timing
            parts.append(aux_html)
    parts.append(_plot_html(output_dir, name))
    return f'<section>{"".join(parts)}</section>', has_timing


def _backend_legend():
    swatches = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{name}</span>'
        for name, color in BACKEND_COLORS.items()
    )
    return f'<div class="legend">{swatches}</div>'


def _render_group(output_dir, names):
    sections = []
    any_timing = False
    for name in names:
        result = _study_section(output_dir, name)
        if result is None:
            continue
        section, has_timing = result
        sections.append(section)
        any_timing = any_timing or has_timing
    return sections, any_timing


def build_html_report(output_dir, title="trust-bench report"):
    """Bundles a report's artefacts into one self-contained HTML page:
    Tier 1's headline artefacts first (each with a caption), then every
    other study grouped by theme (table, caption, and plot where one
    exists), a fixed backend-colour legend, coloured status pills, and
    - only when at least one study measured real timing - a caveat on
    what that timing data does and doesn't support. A study absent from
    output_dir (skipped upstream, e.g. for a coverage gap) is silently
    omitted rather than raising.
    """
    output_dir = Path(output_dir)
    any_timing = False
    body_sections = [_backend_legend()]

    tier1_sections, tier1_timing = _render_group(output_dir, TIER_1)
    any_timing = any_timing or tier1_timing
    if tier1_sections:
        body_sections.append(f'<section><h2>Tier 1: At a glance</h2>{"".join(tier1_sections)}</section>')

    for group_name, study_names in TIER_2_GROUPS:
        group_sections, group_timing = _render_group(output_dir, study_names)
        any_timing = any_timing or group_timing
        if group_sections:
            body_sections.append(f'<section><h2>{group_name}</h2>{"".join(group_sections)}</section>')

    if any_timing:
        body_sections.insert(1, f'<p class="timing-caveat">{TIMING_CAVEAT}</p>')

    body = "\n".join(body_sections)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        f"{_STYLE}</head><body><h1>{title}</h1>{body}</body></html>"
    )


def save_html_report(html, path):
    Path(path).write_text(html)
