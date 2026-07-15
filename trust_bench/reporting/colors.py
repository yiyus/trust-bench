"""Fixed colour assignments so a backend or status reads identically
across every table and plot in a report, rather than depending on
matplotlib's own default cycling or per-render group ordering.
"""

BACKEND_COLORS = {
    "scipy": "#1f77b4",
    "trust-apl": "#d62728",
}

_FALLBACK_BACKEND_COLOR = "#7f7f7f"

STATUS_COLORS = {
    "CONVERGED": "#2ca02c",
    "MAX_ITER": "#ff9800",
    "FAILED": "#d62728",
    "DIVERGED": "#d62728",
    "STALLED": "#ff9800",
    "ERROR": "#d62728",
    "UNSUPPORTED": "#9e9e9e",
}


def backend_color(name):
    return BACKEND_COLORS.get(name, _FALLBACK_BACKEND_COLOR)


def status_color(name):
    return STATUS_COLORS.get(name, _FALLBACK_BACKEND_COLOR)
