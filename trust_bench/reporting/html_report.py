import base64
from pathlib import Path

import pandas as pd

from trust_bench.reporting.colors import BACKEND_COLORS, backend_color, status_color

_STYLE = """
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 0 0 4rem; background: #f6f7f6; color: #1a2420; line-height: 1.5; }
  .hero { max-width: 56rem; margin: 0 auto; padding: 3rem 1.5rem 1rem; }
  .eyebrow { font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase;
             color: #5c6864; margin: 0 0 0.4rem; }
  h1 { font-size: 2rem; margin: 0 0 0.9rem; text-wrap: balance; }
  .lede { font-size: 1.05rem; color: #4d5852; max-width: 44rem; margin: 0; }
  .legend { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-top: 1.2rem; font-size: 0.85rem; }
  .legend-item { display: inline-flex; align-items: center; }
  .swatch { display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 50%; margin-right: 0.4rem; }
  h2 { max-width: 56rem; margin: 2.6rem auto 0.3rem; padding: 1.6rem 1.5rem 0; font-size: 1.3rem;
       border-top: 1px solid #dde3de; }
  h2:first-of-type { border-top: none; padding-top: 0; margin-top: 1.5rem; }
  .tier-intro { max-width: 56rem; margin: 0 auto 1.2rem; padding: 0 1.5rem; color: #4d5852; font-size: 0.95rem; }
  .card { max-width: 56rem; margin: 0 auto 1.2rem; padding: 1.4rem 1.5rem;
          background: #ffffff; border: 1px solid #dde3de; border-radius: 10px; }
  .card h3 { margin: 0 0 0.35rem; font-size: 1.1rem; }
  .caption { color: #4d5852; font-size: 0.9rem; margin: 0 0 1rem; max-width: 48rem; }
  .plot { overflow-x: auto; }
  .plot img, .tablewrap img { max-width: 100%; border-radius: 6px; border: 1px solid #dde3de; }
  .tablewrap { overflow-x: auto; margin-bottom: 0.9rem; }
  table { border-collapse: collapse; font-size: 0.82rem; font-variant-numeric: tabular-nums; width: 100%; }
  th, td { border-bottom: 1px solid #dde3de; padding: 0.4rem 0.6rem; text-align: right; white-space: nowrap; }
  th { color: #5c6864; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
  th:first-child, td:first-child { text-align: left; }
  .pill { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; color: white; font-size: 0.74rem;
          font-weight: 600; }
  .backend-cell { font-weight: 600; font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
                  font-size: 0.78rem; }
  .timing-caveat { font-size: 0.8rem; color: #4d5852; font-style: italic; max-width: 56rem; margin: 0 auto 1.5rem;
                   padding: 0 1.5rem; }
  @media (prefers-color-scheme: dark) {
    body { background: #14181a; color: #e7ede9; }
    .card { background: #1b201f; }
    th, td, h2 { border-color: #2b3733; }
    .plot img, .tablewrap img { border-color: #2b3733; }
    .eyebrow, .lede, .caption, .tier-intro, .timing-caveat, th { color: #a9b3ad; }
  }
</style>
"""

HERO_EYEBROW = "trust-bench &middot; comparison report"

HERO_LEDE = (
    "Three headline artefacts answer the only two questions that matter: do backends "
    "agree on ordinary problems, and where does agreement stop. Everything below is "
    "supporting detail, grouped by what it's testing rather than by filename."
)

# (status used to look up its colour, label shown in the legend) - failure
# modes that read the same way to a reader share one legend entry rather
# than one per RunStatus member.
STATUS_LEGEND = [
    ("CONVERGED", "CONVERGED"),
    ("MAX_ITER", "MAX_ITER / STALLED"),
    ("FAILED", "FAILED / DIVERGED / ERROR"),
    ("UNSUPPORTED", "UNSUPPORTED"),
]

# Tier 1: headline artefacts a reader should see first, before any
# per-study detail - the "at a glance" answer to "do these backends
# agree, and where does each one's frontier lie". Plot only: the point
# is the visual read, not the underlying row-by-row table.
TIER_1 = ["parity_scatter", "capability_frontier", "capability_matrix"]

# Tier 2: every other study, grouped by what question it answers, in
# display order.
TIER_2_GROUPS = [
    ("Common problems", ["baseline", "typical", "derivative_source"]),
    ("Difficulty axes", ["ill_conditioning", "scaling", "large_residual", "dimensionality"]),
    ("Special capabilities", ["robust_loss", "bounded"]),
]

TIER_2_INTROS = {
    "Common problems": (
        "The floor everything must clear: standard test problems at their standard "
        "starting points, no adversarial conditions."
    ),
    "Difficulty axes": (
        "Each of these feeds a panel in the capability frontier above - the table is "
        "the detail behind that chart's transition point for this axis."
    ),
    "Special capabilities": (
        "Axes where the backends' feature sets genuinely differ, not just their "
        "numerical behaviour on the same problem."
    ),
}

# Descriptive headings, distinct from the study's own filename/registry key.
TITLES = {
    "compare": "Longitudinal comparison: regression vs drift",
    "parity_scatter": "Parity: do backends agree?",
    "capability_frontier": "Capability frontier: where does agreement stop?",
    "capability_matrix": "Capability matrix: declared vs measured",
    "baseline": "Baseline correctness",
    "typical": "Typical fitting problems",
    "derivative_source": "Derivative source",
    "ill_conditioning": "Ill-conditioning",
    "scaling": "Variable scaling",
    "large_residual": "Large residual (LM vs Newton)",
    "dimensionality": "Dimensionality",
    "robust_loss": "Robust loss / outliers",
    "bounded": "Bounded / constrained",
}

# One-line captions, adapted from docs/plans/trust-bench.md's own study
# descriptions. A study without an entry here (there shouldn't be one,
# outside of scalar_cost, which this report never renders) gets no
# caption rather than a KeyError.
CAPTIONS = {
    "compare": (
        "Tier-1 metric changes since the baseline run are regressions; Tier-3 "
        "timing-only changes are drift, expected across machines and versions - "
        "see the table's full baseline/candidate provenance for attribution."
    ),
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
    "parity_scatter": "A point on the diagonal means the two backends land on the same answer, no backend framed as better.",
    "capability_frontier": (
        "The value at any single point matters less than the transition - where a "
        "backend's line departs from a flat, converged baseline."
    ),
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


def _render_cell(column, value):
    if column == "status":
        return f'<span class="pill" style="background:{status_color(value)}">{value}</span>'
    if column == "backend":
        return f'<span class="backend-cell" style="color:{backend_color(value)}">{value}</span>'
    return value


def _render_table(df):
    df = df.copy()
    has_timing = "timing_median" in df.columns and df["timing_median"].notna().any()
    if "timing_median" in df.columns:
        df["timing"] = [_format_timing(median, mad) for median, mad in zip(df["timing_median"], df["timing_mad"])]
        df = df.drop(columns=[field for field in _TIMING_FIELDS if field in df.columns])
    for column in ("status", "backend"):
        if column in df.columns:
            df[column] = df[column].map(lambda value, column=column: _render_cell(column, value))
    table_html = df.to_html(index=False, na_rep="", escape=False)
    return f'<div class="tablewrap">{table_html}</div>', has_timing


def _plot_html(output_dir, name):
    png_path = output_dir / f"{name}.png"
    if not png_path.exists():
        return ""
    encoded = base64.b64encode(png_path.read_bytes()).decode()
    return f'<div class="plot"><img src="data:image/png;base64,{encoded}" alt="{name}"></div>'


def _card(name, body_parts):
    title = TITLES.get(name, name)
    caption = CAPTIONS.get(name, "")
    return f'<section class="card"><h3>{title}</h3><p class="caption">{caption}</p>{"".join(body_parts)}</section>'


def _headline_section(output_dir, name):
    plot_html = _plot_html(output_dir, name)
    if not plot_html:
        return None
    return _card(name, [plot_html]), False


def _study_section(output_dir, name):
    csv_path = output_dir / f"{name}.csv"
    if not csv_path.exists():
        return None
    table_html, has_timing = _render_table(pd.read_csv(csv_path))
    parts = [table_html]
    for aux_name in AUX_TABLES.get(name, []):
        aux_path = output_dir / f"{aux_name}.csv"
        if aux_path.exists():
            aux_html, aux_timing = _render_table(pd.read_csv(aux_path))
            has_timing = has_timing or aux_timing
            parts.append(aux_html)
    parts.append(_plot_html(output_dir, name))
    return _card(name, parts), has_timing


def _render_group(output_dir, names, section_builder):
    sections = []
    any_timing = False
    for name in names:
        result = section_builder(output_dir, name)
        if result is None:
            continue
        section, has_timing = result
        sections.append(section)
        any_timing = any_timing or has_timing
    return sections, any_timing


def _hero(title):
    legend_items = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>{name}</span>'
        for name, color in BACKEND_COLORS.items()
    )
    legend_items += "".join(
        f'<span class="legend-item"><span class="pill" style="background:{status_color(status)}">{label}</span></span>'
        for status, label in STATUS_LEGEND
    )
    return (
        f'<section class="hero"><p class="eyebrow">{HERO_EYEBROW}</p><h1>{title}</h1>'
        f'<p class="lede">{HERO_LEDE}</p><div class="legend">{legend_items}</div></section>'
    )


def build_html_report(output_dir, title="trust-bench report"):
    """Bundles a report's artefacts into one self-contained HTML page: a
    hero with a fixed backend/status legend, Tier 1's headline plots
    first (each with a caption), then every other study grouped by theme
    (a one-line theme intro, then each study's own table, caption, and
    plot where one exists), and - only when at least one study measured
    real timing - a caveat on what that timing data does and doesn't
    support. A study absent from output_dir (skipped upstream, e.g. for
    a coverage gap) is silently omitted rather than raising.
    """
    output_dir = Path(output_dir)
    any_timing = False
    body_sections = [_hero(title)]

    comparison_result = _study_section(output_dir, "compare")
    if comparison_result is not None:
        comparison_section, comparison_timing = comparison_result
        any_timing = any_timing or comparison_timing
        body_sections.append(f"<h2>Longitudinal comparison</h2>{comparison_section}")

    tier1_sections, tier1_timing = _render_group(output_dir, TIER_1, _headline_section)
    any_timing = any_timing or tier1_timing
    if tier1_sections:
        body_sections.append(f'<h2>At a glance</h2>{"".join(tier1_sections)}')

    for group_name, study_names in TIER_2_GROUPS:
        group_sections, group_timing = _render_group(output_dir, study_names, _study_section)
        any_timing = any_timing or group_timing
        if group_sections:
            intro = TIER_2_INTROS.get(group_name, "")
            body_sections.append(
                f'<h2>{group_name}</h2><p class="tier-intro">{intro}</p>{"".join(group_sections)}'
            )

    if any_timing:
        body_sections.append(f'<p class="timing-caveat">{TIMING_CAVEAT}</p>')

    body = "\n".join(body_sections)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>{_STYLE}</head><body>{body}</body></html>"


def save_html_report(html, path):
    Path(path).write_text(html)
