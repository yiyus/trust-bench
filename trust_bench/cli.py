import argparse
import sys
from pathlib import Path

import pandas as pd

from trust_bench.backends import BACKENDS
from trust_bench.backends.apl_backend import APLBackend
from trust_bench.backends.scipy_backend import SciPyBackend
from trust_bench.reporting.capability_matrix import derive_matrix
from trust_bench.reporting.cross_study import frontier_panels, parity_frame
from trust_bench.reporting.html_report import build_html_report, save_html_report
from trust_bench.reporting.plots import (
    plot_capability_frontier,
    plot_capability_matrix,
    plot_metric_by_category,
    plot_metric_vs_sweep,
    plot_parity_scatter,
    save_figure,
)
from trust_bench.reporting.tables import NON_RESULT_STATUSES, results_to_dataframe, save_table
from trust_bench.studies import (
    baseline,
    bounded,
    derivative_source,
    dimensionality,
    ill_conditioning,
    large_residual,
    robust_loss,
    scalar_cost,
    scaling,
    typical,
)

AVAILABLE_BACKENDS = {backend.name: backend for backend in [SciPyBackend(), APLBackend()]}


class CoverageError(ValueError):
    """Raised only by _check_backend_coverage - distinct from any other
    ValueError a study's write path might raise (an unsupported
    x_scale/loss/derivative_mode combination, an invalid problem-family
    parameter), so run_report's per-study catch absorbs a genuine, known
    coverage gap and nothing else.
    """


def _check_backend_coverage(df, backends, study):
    """Raise clearly if a selected backend produced zero usable rows for
    this study, rather than let a study's own per-pair skip guard (e.g.
    ill_conditioning.sweep's "method not supported: continue") silently
    produce an empty report table, or a backend whose problem ids the
    study doesn't recognise silently pass coverage on ERROR/UNSUPPORTED
    rows alone.
    """
    rows = df[~df["status"].isin(NON_RESULT_STATUSES)] if "status" in df.columns else df
    missing = {backend.name for backend in backends} - set(rows["backend"])
    if missing:
        raise CoverageError(f"{study}: no results for backend(s) {', '.join(sorted(missing))}")


def _baseline_basin_rate_table(backends):
    return pd.DataFrame(
        dict(problem_id=problem_id, backend=backend, basin_rate=rate)
        for (problem_id, backend), rate in baseline.basin_rates(backends=backends).items()
    )


def _write_baseline(output_dir, backends):
    df = results_to_dataframe(
        baseline.standard_start_results(backends=backends), key_names=["problem_id", "backend"]
    )
    _check_backend_coverage(df, backends, "baseline")
    save_table(df, output_dir / "baseline.csv")

    basin_df = _baseline_basin_rate_table(backends)
    _check_backend_coverage(basin_df, backends, "baseline (basin rates)")
    save_table(basin_df, output_dir / "baseline_basin_rates.csv")


def _large_residual_basin_rate_table(rates):
    return pd.DataFrame(dict(rho=rho, backend=backend, basin_rate=rate) for (rho, backend), rate in rates.items())


def _write_large_residual(output_dir, backends):
    results, rates = large_residual.backend_results(backends=backends)
    df = results_to_dataframe(results, key_names=["rho", "backend"])
    _check_backend_coverage(df, backends, "large_residual")
    save_table(df, output_dir / "large_residual.csv")
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend", logx=True, logy=True)
    save_figure(fig, output_dir / "large_residual.png")

    basin_df = _large_residual_basin_rate_table(rates)
    _check_backend_coverage(basin_df, backends, "large_residual (basin rates)")
    save_table(basin_df, output_dir / "large_residual_basin_rates.csv")


def _write_ill_conditioning(output_dir, backends):
    df = results_to_dataframe(ill_conditioning.sweep(backends=backends), key_names=["kappa", "method", "backend"])
    _check_backend_coverage(df, backends, "ill_conditioning")
    save_table(df, output_dir / "ill_conditioning.csv")
    fig = plot_metric_vs_sweep(
        df,
        x="kappa",
        y="dist_to_opt",
        group=["method", "backend"],
        logx=True,
        logy=True,
        status_col="status",
    )
    save_figure(fig, output_dir / "ill_conditioning.png")


def _robust_loss_table(backends):
    rows = [
        dict(fraction=fraction, loss=loss, backend=backend, distance=distance)
        for (fraction, loss, backend), distance in robust_loss.scipy_loss_precision(backends=backends).items()
    ]
    rows += [
        dict(fraction=fraction, loss=loss, backend=backend, distance=distance)
        for (fraction, loss, backend), distance in robust_loss.trust_loss_precision(backends=backends).items()
    ]
    rows += [
        dict(fraction=fraction, loss="irls_tukey", backend="hand-rolled", distance=distance)
        for fraction, distance in robust_loss.irls_precision().items()
    ]
    return pd.DataFrame(rows)


def _write_robust_loss(output_dir, backends):
    df = _robust_loss_table(backends)
    _check_backend_coverage(df, backends, "robust_loss")
    save_table(df, output_dir / "robust_loss.csv")
    fig = plot_metric_vs_sweep(df, x="fraction", y="distance", group=["loss", "backend"])
    save_figure(fig, output_dir / "robust_loss.png")


def _write_bounded(output_dir, backends):
    df = results_to_dataframe(bounded.sweep(backends=backends), key_names=["scenario", "method", "backend"])
    _check_backend_coverage(df, backends, "bounded")
    save_table(df, output_dir / "bounded.csv")


def _write_scaling(output_dir, backends):
    df = results_to_dataframe(
        scaling.sweep(backends=backends), key_names=["scale", "method", "x_scale", "backend"]
    )
    _check_backend_coverage(df, backends, "scaling")
    save_table(df, output_dir / "scaling.csv")
    fig = plot_metric_vs_sweep(
        df, x="scale", y="grad_norm_final", group=["method", "x_scale", "backend"], logx=True, logy=True
    )
    save_figure(fig, output_dir / "scaling.png")


def _write_dimensionality(output_dir, backends):
    df = results_to_dataframe(dimensionality.sweep(backends=backends), key_names=["n", "method", "backend"])
    _check_backend_coverage(df, backends, "dimensionality")
    save_table(df, output_dir / "dimensionality.csv")
    fig = plot_metric_vs_sweep(df, x="n", y="dist_to_opt", group=["method", "backend"], logx=True)
    save_figure(fig, output_dir / "dimensionality.png")


def _write_derivative_source(output_dir, backends):
    df = results_to_dataframe(
        derivative_source.sweep(backends=backends), key_names=["problem_id", "method", "mode", "backend"]
    )
    _check_backend_coverage(df, backends, "derivative_source")
    save_table(df, output_dir / "derivative_source.csv")


def _write_scalar_cost(output_dir, backends):
    df = results_to_dataframe(scalar_cost.sweep(backends=backends), key_names=["problem_id", "method", "backend"])
    _check_backend_coverage(df, backends, "scalar_cost")
    save_table(df, output_dir / "scalar_cost.csv")


def _write_capability_matrix(output_dir, backends):
    df = derive_matrix(backends=backends)
    _check_backend_coverage(df, backends, "capability_matrix")
    save_table(df, output_dir / "capability_matrix.csv")
    fig = plot_capability_matrix(df)
    save_figure(fig, output_dir / "capability_matrix.png")


def _write_parity_scatter(output_dir, backends):
    df = parity_frame(backends=backends)
    save_table(df, output_dir / "parity_scatter.csv")
    b1, b2 = backends[0].name, backends[1].name
    fig = plot_parity_scatter(
        df, x=f"dist_to_opt_{b1}", y=f"dist_to_opt_{b2}", converged_col="converged", group="study"
    )
    save_figure(fig, output_dir / "parity_scatter.png")


def _write_capability_frontier(output_dir, backends):
    panels = frontier_panels(backends=backends)
    combined = pd.concat([df.assign(panel=name) for name, (df, _, _) in panels.items()], ignore_index=True)
    _check_backend_coverage(combined, backends, "capability_frontier")
    save_table(combined, output_dir / "capability_frontier.csv")
    fig = plot_capability_frontier(panels)
    save_figure(fig, output_dir / "capability_frontier.png")


def _write_typical(output_dir, backends):
    df = results_to_dataframe(typical.sweep(backends=backends), key_names=["problem_id", "method", "backend"])
    _check_backend_coverage(df, backends, "typical")
    save_table(df, output_dir / "typical.csv")
    fig = plot_metric_by_category(
        df, category="problem_id", y="dist_to_opt", group=["method", "backend"], logy=True, status_col="status"
    )
    save_figure(fig, output_dir / "typical.png")


STUDIES = {
    "baseline": _write_baseline,
    "large_residual": _write_large_residual,
    "ill_conditioning": _write_ill_conditioning,
    "robust_loss": _write_robust_loss,
    "bounded": _write_bounded,
    "scaling": _write_scaling,
    "dimensionality": _write_dimensionality,
    "derivative_source": _write_derivative_source,
    "scalar_cost": _write_scalar_cost,
    "capability_matrix": _write_capability_matrix,
    "typical": _write_typical,
    "parity_scatter": _write_parity_scatter,
    "capability_frontier": _write_capability_frontier,
}

# dimensionality sweeps up to n=1000 with a dense-Hessian method
# (Section 9 item 7's own point), which costs tens of seconds per run;
# every other study/artefact here runs in a few seconds at most.
# capability_frontier's own dimensionality panel repeats that same
# n=1000 sweep (for a single method, so cheaper, but still the slowest
# thing this artefact computes).
SLOW_STUDIES = frozenset({"dimensionality", "capability_frontier"})

# parity_scatter is inherently a two-backend comparison (a pairwise
# scipy-vs-trust-apl pivot, unlike capability_frontier's one-line-per-
# backend panels, which work fine with just one); auto-excluded from
# the default "run everything" selection when fewer than two backends
# are selected, matching a plain `trust-bench report`'s scipy-only
# default, which must keep working without dyalogscript installed.
# Naming it explicitly via --only still runs it, where a clear error
# (from parity_frame's own backend-count check) is the right response
# to an impossible request, rather than a silent skip.
MULTI_BACKEND_STUDIES = frozenset({"parity_scatter"})


def _select_studies(only=None, skip=None, skip_slow=False, n_backends=1):
    selected = set(only) if only is not None else set(STUDIES)
    if skip is not None:
        selected -= set(skip)
    if skip_slow:
        selected -= SLOW_STUDIES
    if only is None and n_backends < 2:
        selected -= MULTI_BACKEND_STUDIES

    unknown = selected - set(STUDIES)
    if unknown:
        raise ValueError(f"Unknown study/studies: {', '.join(sorted(unknown))}")
    return selected


def _select_backends(names=None):
    if names is None:
        return BACKENDS

    unknown = set(names) - set(AVAILABLE_BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backend/backends: {', '.join(sorted(unknown))}")
    return [AVAILABLE_BACKENDS[name] for name in names]


def run_report(output_dir, only=None, skip=None, skip_slow=False, html=False, backends=None):
    """Runs every selected study, writing each one's artefacts to
    output_dir. A study whose selected backend(s) have a known,
    permanent coverage gap (e.g. trust-apl has no evaluator for
    scalar_cost's Jacobian-free scalar objectives at all) is skipped,
    not fatal to the rest of the report - _check_backend_coverage's own
    CoverageError is caught per study and collected into the returned
    `skipped` mapping instead of aborting. If every selected study fails
    this way there is nothing to report at all, a genuine failure, so
    that case still raises.
    """
    selected_backends = _select_backends(backends)
    selected = _select_studies(only=only, skip=skip, skip_slow=skip_slow, n_backends=len(selected_backends))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = {}
    for name in sorted(selected):
        try:
            STUDIES[name](output_dir, selected_backends)
        except CoverageError as error:
            skipped[name] = str(error)

    if skipped and len(skipped) == len(selected):
        reasons = "; ".join(f"{name}: {message}" for name, message in sorted(skipped.items()))
        raise ValueError(f"every selected study failed - {reasons}")

    if html:
        save_html_report(build_html_report(output_dir), output_dir / "report.html")

    return output_dir, skipped


def build_parser():
    parser = argparse.ArgumentParser(
        prog="trust-bench",
        description="Optimisation-solver comparison harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="Run the capability studies and write tables, plots, and the capability matrix.",
        description=(
            "Runs the registered studies end-to-end and writes their comparison "
            "tables (and, where applicable, a plot) plus the capability matrix "
            "to the output directory."
        ),
    )
    report_parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write the generated artefacts to (default: %(default)s).",
    )
    selection = report_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only",
        nargs="+",
        choices=sorted(STUDIES),
        metavar="STUDY",
        help="Run only these studies/artefacts, instead of all of them.",
    )
    selection.add_argument(
        "--skip",
        nargs="+",
        choices=sorted(STUDIES),
        metavar="STUDY",
        help="Run every study/artefact except these.",
    )
    report_parser.add_argument(
        "--skip-slow",
        action="store_true",
        help=f"Skip studies that take noticeably longer to run ({', '.join(sorted(SLOW_STUDIES))}).",
    )
    report_parser.add_argument(
        "--backends",
        nargs="+",
        choices=sorted(AVAILABLE_BACKENDS),
        metavar="BACKEND",
        help="Run against these backends instead of the default (scipy).",
    )
    report_parser.add_argument(
        "--html",
        action="store_true",
        help="Also write report.html, bundling every table and plot into one self-contained page.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "report":
        output_dir, skipped = run_report(
            args.output_dir,
            only=args.only,
            skip=args.skip,
            skip_slow=args.skip_slow,
            html=args.html,
            backends=args.backends,
        )
        for name, message in sorted(skipped.items()):
            print(f"Skipped {name}: {message}", file=sys.stderr)
        print(f"Report artefacts written to {output_dir}")


if __name__ == "__main__":
    main()
