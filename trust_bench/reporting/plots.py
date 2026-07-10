import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def plot_metric_vs_sweep(df, x, y, group=None, logx=False, logy=False):
    fig, ax = plt.subplots()
    if group is None:
        ax.plot(df[x], df[y], marker="o")
    else:
        for name, subset in df.groupby(group):
            ax.plot(subset[x], subset[y], marker="o", label=str(name))
        ax.legend()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig


def save_figure(fig, path):
    fig.savefig(path)
    plt.close(fig)
