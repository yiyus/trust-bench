import matplotlib

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
