import argparse
import sys
from pathlib import Path

import pandas as pd

from trust_bench.reporting.capability_matrix import derive_matrix
from trust_bench.reporting.plots import plot_metric_vs_sweep, save_figure
from trust_bench.reporting.tables import results_to_dataframe, save_table
from trust_bench.studies import (
    baseline,
    bounded,
    derivative_source,
    dimensionality,
    ill_conditioning,
    large_residual,
    robust_loss,
    scaling,
)


def _write_baseline(output_dir):
    df = results_to_dataframe(baseline.standard_start_results(), key_names=["problem_id", "backend"])
    save_table(df, output_dir / "baseline.csv")


def _write_large_residual(output_dir):
    results, _ = large_residual.backend_results()
    df = results_to_dataframe(results, key_names=["rho", "backend"])
    save_table(df, output_dir / "large_residual.csv")
    fig = plot_metric_vs_sweep(df, x="rho", y="grad_norm_final", group="backend", logx=True, logy=True)
    save_figure(fig, output_dir / "large_residual.png")


def _write_ill_conditioning(output_dir):
    df = results_to_dataframe(ill_conditioning.sweep(), key_names=["kappa", "method", "backend"])
    save_table(df, output_dir / "ill_conditioning.csv")


def _robust_loss_table():
    rows = [
        dict(fraction=fraction, loss=loss, backend=backend, distance=distance)
        for (fraction, loss, backend), distance in robust_loss.scipy_loss_precision().items()
    ]
    rows += [
        dict(fraction=fraction, loss="irls_tukey", backend="hand-rolled", distance=distance)
        for fraction, distance in robust_loss.irls_precision().items()
    ]
    return pd.DataFrame(rows)


def _write_robust_loss(output_dir):
    save_table(_robust_loss_table(), output_dir / "robust_loss.csv")


def _write_bounded(output_dir):
    df = results_to_dataframe(bounded.sweep(), key_names=["scenario", "method", "backend"])
    save_table(df, output_dir / "bounded.csv")


def _write_scaling(output_dir):
    df = results_to_dataframe(scaling.sweep(), key_names=["scale", "method", "x_scale", "backend"])
    save_table(df, output_dir / "scaling.csv")


def _write_dimensionality(output_dir):
    df = results_to_dataframe(dimensionality.sweep(), key_names=["n", "method", "backend"])
    save_table(df, output_dir / "dimensionality.csv")


def _write_derivative_source(output_dir):
    df = results_to_dataframe(
        derivative_source.sweep(), key_names=["problem_id", "method", "mode", "backend"]
    )
    save_table(df, output_dir / "derivative_source.csv")


def _write_capability_matrix(output_dir):
    save_table(derive_matrix(), output_dir / "capability_matrix.csv")


STUDIES = {
    "baseline": _write_baseline,
    "large_residual": _write_large_residual,
    "ill_conditioning": _write_ill_conditioning,
    "robust_loss": _write_robust_loss,
    "bounded": _write_bounded,
    "scaling": _write_scaling,
    "dimensionality": _write_dimensionality,
    "derivative_source": _write_derivative_source,
    "capability_matrix": _write_capability_matrix,
}

# dimensionality sweeps up to n=1000 with a dense-Hessian method
# (Section 9 item 7's own point), which costs tens of seconds per run;
# every other study/artefact here runs in a few seconds at most.
SLOW_STUDIES = frozenset({"dimensionality"})


def _select_studies(only=None, skip=None, skip_slow=False):
    selected = set(only) if only is not None else set(STUDIES)
    if skip is not None:
        selected -= set(skip)
    if skip_slow:
        selected -= SLOW_STUDIES

    unknown = selected - set(STUDIES)
    if unknown:
        raise ValueError(f"Unknown study/studies: {', '.join(sorted(unknown))}")
    return selected


def run_report(output_dir, only=None, skip=None, skip_slow=False):
    selected = _select_studies(only=only, skip=skip, skip_slow=skip_slow)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(selected):
        STUDIES[name](output_dir)
    return output_dir


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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "report":
        output_dir = run_report(
            args.output_dir, only=args.only, skip=args.skip, skip_slow=args.skip_slow
        )
        print(f"Report artefacts written to {output_dir}")


if __name__ == "__main__":
    main()
