import numpy as np
import pandas as pd

from trust_bench.backends import BACKENDS
from trust_bench.core.config import RunConfig
from trust_bench.core.result import RunStatus
from trust_bench.core.runner import run
from trust_bench.problems.families import outliers
from trust_bench.reporting.tables import results_to_dataframe
from trust_bench.studies import bounded, dimensionality, ill_conditioning, large_residual, scaling, typical
from trust_bench.studies import baseline as baseline_study
from trust_bench.studies.robust_loss import FRACTIONS

# A log-scale display floor for dist_to_opt, not a real precision claim:
# several problems reach dist_to_opt == 0.0 exactly, which log(0) can't
# plot. Well below the smallest nonzero value measured anywhere in this
# project (methodology.md's own measured floor is ~1e-14).
_DIST_TO_OPT_FLOOR = 1e-16

# One representative method per difficulty sweep, chosen to show the
# clearest trust-apl-vs-scipy transition documented in
# docs/methodology.md, rather than every method the study itself
# sweeps (which plot_metric_vs_sweep already shows, per-study).
_ILL_CONDITIONING_METHOD = "trust-exact"
_SCALING_METHOD = "lm"
_DIMENSIONALITY_METHOD = "BFGS"

# scipy and trust-apl declare no common method name for bounded
# least-squares (scipy's own "lm" declares bounds=False; trust-apl's
# only bounds-capable method is its Coleman-Li-scaled "lm"), so a
# (scenario, method) pivot never finds a matching pair - mirroring
# robust_loss.py's own _METHOD_FOR_BACKEND, pick each backend's natural
# bounds-capable method instead of requiring the same name.
_BOUNDED_METHOD_FOR_BACKEND = {"scipy": "trf", "trust-apl": "lm"}


def _pivot_by_backend(df, index_cols, backends, metric="dist_to_opt"):
    """Wide-form frame with one {metric}_{backend.name} and
    converged_{backend.name} column pair per backend, dropping any row
    missing a value for either - a parity comparison needs both sides.
    """
    if len(backends) != 2:
        raise ValueError(f"parity comparison needs exactly two backends, got {[b.name for b in backends]}")

    non_error = df[df["status"] != "ERROR"]
    missing = {b.name for b in backends} - set(non_error["backend"])
    if missing:
        raise ValueError(f"parity_scatter: no results for backend(s) {', '.join(sorted(missing))}")

    df = df.copy()
    df[metric] = df[metric].astype(float).clip(lower=_DIST_TO_OPT_FLOOR)
    df["converged"] = df["status"] == RunStatus.CONVERGED.value
    wide = df.pivot(index=index_cols, columns="backend", values=[metric, "converged"])
    wide.columns = [f"{field}_{backend_name}" for field, backend_name in wide.columns]
    wide = wide.reset_index()
    value_cols = [f"{metric}_{b.name}" for b in backends]
    return wide.dropna(subset=value_cols)


def parity_frame(backends=BACKENDS):
    """Pools baseline/typical/bounded (the common, non-adversarial
    studies already shaped like results_to_dataframe: dist_to_opt and
    status per row) into one long frame, one row per (study, instance,
    backend pair), for the parity scatter. robust_loss is excluded: its
    own precision tables report distance to the true parameters, not
    dist_to_opt, and don't share this shape.
    """
    if len(backends) != 2:
        raise ValueError(f"parity comparison needs exactly two backends, got {[b.name for b in backends]}")
    b1, b2 = backends[0].name, backends[1].name
    frames = []

    baseline_df = results_to_dataframe(
        baseline_study.standard_start_results(backends=backends), key_names=["problem_id", "backend"]
    )
    wide = _pivot_by_backend(baseline_df, ["problem_id"], backends)
    wide["study"] = "baseline"
    frames.append(wide)

    typical_df = results_to_dataframe(typical.sweep(backends=backends), key_names=["problem_id", "method", "backend"])
    wide = _pivot_by_backend(typical_df, ["problem_id", "method"], backends)
    wide["study"] = "typical"
    frames.append(wide)

    bounded_df = results_to_dataframe(bounded.sweep(backends=backends), key_names=["scenario", "method", "backend"])
    bounded_df = bounded_df[
        bounded_df.apply(lambda row: _BOUNDED_METHOD_FOR_BACKEND.get(row["backend"]) == row["method"], axis=1)
    ]
    wide = _pivot_by_backend(bounded_df, ["scenario"], backends)
    wide["study"] = "bounded"
    frames.append(wide)

    pooled = pd.concat(frames, ignore_index=True)
    pooled["converged"] = pooled[f"converged_{b1}"] & pooled[f"converged_{b2}"]
    return pooled


def _robust_loss_linear_frame(backends):
    """Distance from the true (uncorrupted) parameters per outlier
    fraction and backend, for the one loss ("linear") every backend
    already declares support for - unlike robust_loss.py's own
    scipy_loss_precision/trust_loss_precision, which sweep disjoint
    loss vocabularies and share no common loss to compare directly.

    Distance to TRUE_PARAMETERS, not result.dist_to_opt: a "linear"-
    loss fit converges to the corrupted data's own L2 minimum (this
    family's registered optimum) reliably regardless of contamination,
    so dist_to_opt only checks solver correctness - it says nothing
    about the robustness this study exists to measure, unlike the
    distance to the true, uncorrupted parameters robust_loss.py's own
    precision functions already report.
    """
    rows = []
    for fraction in FRACTIONS:
        problem = outliers.make(fraction)
        for backend in backends:
            result = run(problem, backend, "lm", "standard", RunConfig(max_iter=200, loss="linear"))
            x_final = np.asarray(result.x_final, dtype=float)
            distance = float(np.linalg.norm(x_final - outliers.TRUE_PARAMETERS))
            rows.append(dict(fraction=fraction, backend=backend.name, dist_to_opt=distance))
    return pd.DataFrame(rows)


def frontier_panels(backends=BACKENDS):
    """One (df, x, y) entry per existing difficulty sweep, keyed by a
    short display name, for the capability-frontier chart. Each frame
    is restricted to a single representative method (see the module
    constants above) so every panel is a clean backend-vs-backend line,
    matching plot_capability_frontier's own one-line-per-backend shape.
    """
    ill_conditioning_df = results_to_dataframe(
        ill_conditioning.sweep(methods=[_ILL_CONDITIONING_METHOD], backends=backends),
        key_names=["kappa", "method", "backend"],
    )
    scaling_df = results_to_dataframe(
        scaling.sweep(methods=[_SCALING_METHOD], x_scales=[None], backends=backends),
        key_names=["scale", "method", "x_scale", "backend"],
    )
    dimensionality_df = results_to_dataframe(
        dimensionality.sweep(methods=[_DIMENSIONALITY_METHOD], backends=backends),
        key_names=["n", "method", "backend"],
    )
    large_residual_results, _ = large_residual.backend_results(backends=backends)
    large_residual_df = results_to_dataframe(large_residual_results, key_names=["rho", "backend"])
    robust_loss_df = _robust_loss_linear_frame(backends)

    panels = {
        "ill_conditioning": (ill_conditioning_df, "kappa", "dist_to_opt"),
        "scaling": (scaling_df, "scale", "grad_norm_final"),
        "dimensionality": (dimensionality_df, "n", "dist_to_opt"),
        "large_residual": (large_residual_df, "rho", "grad_norm_final"),
        "robust_loss": (robust_loss_df, "fraction", "dist_to_opt"),
    }
    # An exact 0.0 (e.g. scipy's lm reaching machine-precision zero
    # grad_norm_final at every swept scale) can't be log-scaled; floor
    # it to the same display-only value parity_frame uses, rather than
    # let the panel silently disappear or warn.
    for df, _, y in panels.values():
        df[y] = df[y].astype(float).clip(lower=_DIST_TO_OPT_FLOOR)
    return panels
