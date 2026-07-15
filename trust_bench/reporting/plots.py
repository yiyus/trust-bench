import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from trust_bench.reporting.colors import backend_color

_CAPABILITY_CATEGORIES = ["agree", "declared-only", "disagree"]
_CAPABILITY_COLORS = ["#4caf50", "#ff9800", "#f44336"]


def _group_label(name):
    if isinstance(name, tuple):
        return "/".join(str(value) for value in name)
    return str(name)


def _backend_from_group(group, name):
    """The backend name for a groupby key, when "backend" is part of
    group - a bare column name or the matching position in a tuple key
    for a multi-column group - or None otherwise, so callers can apply
    the fixed backend colour only where backend identity is actually
    part of what's being plotted.
    """
    if group == "backend":
        return name
    if isinstance(group, list) and "backend" in group:
        return name[group.index("backend")]
    return None


def plot_metric_vs_sweep(df, x, y, group=None, logx=False, logy=False, status_col=None):
    """group may be a single column name or a list of column names (one
    line per unique combination). If status_col is given, a point whose
    value there isn't "CONVERGED" is additionally drawn as a hollow
    marker in its group's own colour, on top of the regular line, so a
    stalled or failed point is visually distinguishable without being
    dropped from the trend.
    """
    fig, ax = plt.subplots()
    if group is None:
        ax.plot(df[x], df[y], marker="o")
    else:
        for name, subset in df.groupby(group):
            subset = subset.sort_values(x)
            backend_name = _backend_from_group(group, name)
            color = backend_color(backend_name) if backend_name is not None else None
            (line,) = ax.plot(subset[x], subset[y], marker="o", label=_group_label(name), color=color)
            if status_col is not None:
                stalled = subset[subset[status_col] != "CONVERGED"]
                if len(stalled):
                    ax.plot(
                        stalled[x],
                        stalled[y],
                        linestyle="none",
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor=line.get_color(),
                        markersize=9,
                    )
        ax.legend()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig


def plot_metric_by_category(df, category, y, group, logy=False, status_col=None):
    """Grouped bar chart: one cluster per unique value of category, one
    bar per unique combination of group columns within each cluster.
    For categorical, non-sweep data (e.g. the typical study's one
    problem x method x backend per row, with no continuous axis to
    plot against), unlike plot_metric_vs_sweep's line-per-sweep-value
    shape. A bar whose status_col value isn't "CONVERGED" is hatched,
    mirroring plot_metric_vs_sweep's own non-CONVERGED marker.
    """
    df = df.copy()
    df["_group"] = df[group].astype(str).agg("/".join, axis=1) if isinstance(group, list) else df[group].astype(str)
    categories = sorted(df[category].unique())
    groups = sorted(df["_group"].unique())

    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 2.2), 4.5))
    x = np.arange(len(categories))
    width = 0.8 / max(len(groups), 1)
    for i, group_name in enumerate(groups):
        subset = df[df["_group"] == group_name].set_index(category)
        values = subset[y].reindex(categories)
        backend_name = _backend_from_group(group, group_name.split("/") if isinstance(group, list) else group_name)
        color = backend_color(backend_name) if backend_name is not None else None
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=group_name, color=color)
        if status_col is not None:
            statuses = subset[status_col].reindex(categories)
            for bar, status in zip(bars, statuses):
                if status != "CONVERGED":
                    bar.set_hatch("//")
                    bar.set_edgecolor("black")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(y)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_parity_scatter(df, x, y, converged_col=None, group=None):
    """Log-log scatter of x vs y with a y=x reference line: a tight
    cloud along the diagonal is the "these two backends agree" claim in
    one glance, no backend framed as better. A row where converged_col
    is False is drawn with a distinct marker ("x") instead of the
    default filled circle, in its group's own colour, so a
    non-convergent outlier stays visible as the exception it is rather
    than being dropped or mistaken for an ordinary point.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    groups = df.groupby(group) if group is not None else [(None, df)]
    for name, subset in groups:
        converged = subset[converged_col] if converged_col is not None else pd.Series(True, index=subset.index)
        good = subset[converged]
        bad = subset[~converged]
        label = _group_label(name) if group is not None else None
        (line,) = ax.plot(good[x], good[y], marker="o", linestyle="none", label=label)
        if len(bad):
            ax.plot(bad[x], bad[y], marker="x", linestyle="none", color=line.get_color(), markersize=9)

    lo = min(df[x].min(), df[y].min())
    hi = max(df[x].max(), df[y].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_capability_frontier(panels):
    """Small multiples, one panel per (name, (df, x, y)) entry: x and y
    both log-scaled, one line per unique "backend" value. The point is
    the transition - where a backend's line departs from a flat,
    converged baseline and starts climbing (or stops appearing at all)
    - not the value at any single x, so panels share no y-axis and are
    read independently.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, (df, x, y)) in zip(axes, panels.items()):
        for backend_name, subset in df.groupby("backend"):
            subset = subset.sort_values(x)
            ax.plot(subset[x], subset[y], marker="o", label=backend_name, color=backend_color(backend_name))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _capability_category(row):
    if row["agrees"]:
        return "agree"
    if row["declared"] and not row["measured"]:
        return "declared-only"
    return "disagree"


def plot_capability_matrix(df):
    """Rows are capability fields (e.g. "bounds", "analytic_hessian"),
    columns are backend/method pairs, cells are "agree" (declared and
    measured match), "declared-only" (claimed but not confirmed
    working), or "disagree" (the other way round: working despite not
    being declared).
    """
    df = df.copy()
    df["column"] = df["backend"] + "/" + df["method"]
    df["category"] = df.apply(_capability_category, axis=1)
    pivot = df.pivot(index="field", columns="column", values="category")
    codes = pivot.map(_CAPABILITY_CATEGORIES.index)

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 0.9), max(2, len(pivot.index) * 0.7)))
    ax.imshow(codes.values, cmap=ListedColormap(_CAPABILITY_COLORS), vmin=0, vmax=len(_CAPABILITY_CATEGORIES) - 1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, field in enumerate(pivot.index):
        for j, column in enumerate(pivot.columns):
            ax.text(j, i, pivot.loc[field, column], ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    return fig


def save_figure(fig, path):
    fig.savefig(path)
    plt.close(fig)
