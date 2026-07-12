import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

_CAPABILITY_CATEGORIES = ["agree", "declared-only", "disagree"]
_CAPABILITY_COLORS = ["#4caf50", "#ff9800", "#f44336"]


def _group_label(name):
    if isinstance(name, tuple):
        return "/".join(str(value) for value in name)
    return str(name)


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
            (line,) = ax.plot(subset[x], subset[y], marker="o", label=_group_label(name))
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
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=group_name)
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
