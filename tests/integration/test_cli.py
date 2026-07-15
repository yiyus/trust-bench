import shutil

import matplotlib
import pandas as pd
import pytest

from trust_bench.backends import BACKENDS
from trust_bench.cli import (
    AVAILABLE_BACKENDS,
    MULTI_BACKEND_STUDIES,
    SLOW_STUDIES,
    STUDIES,
    _select_backends,
    _select_studies,
    build_parser,
    main,
    run_report,
)
from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.problems import CANONICAL_PROBLEMS
from trust_bench.studies.large_residual import RHOS as LARGE_RESIDUAL_RHOS


class _AlwaysErrorsBackend(Backend):
    """Declares support for "lm" but never actually solves anything: every
    call reports RunStatus.ERROR, regardless of problem. Models a backend
    whose capabilities are declared but whose harness cannot recognise a
    given study's problem ids - the scenario _check_backend_coverage
    exists to catch - without depending on which real problem families a
    real backend happens to have ported at any given time.
    """

    name = "always-errors"

    def capabilities(self):
        return Capabilities(
            methods={
                "lm": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset({"linear"}),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                )
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=None,
            cost_final=None,
            dist_to_opt=None,
            cost_gap=None,
            grad_norm_final=None,
            status=RunStatus.ERROR,
            n_iter=None,
            n_feval=None,
            n_jeval=None,
            n_heval=None,
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp="2026-01-01T00:00:00Z",
        )


class _LmOnlyBackend(Backend):
    """Genuinely solves lm-based studies (baseline) but always reports
    ERROR for BFGS/L-BFGS-B (scalar_cost's own methods) - models
    trust-apl's real, permanent scalar_cost gap without depending on
    dyalogscript being installed.
    """

    name = "lm-only"

    def capabilities(self):
        return Capabilities(
            methods={
                "lm": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset({"linear"}),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                ),
                "BFGS": MethodCapabilities(
                    kind="scalar",
                    losses=frozenset(),
                    bounds=False,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                ),
                "L-BFGS-B": MethodCapabilities(
                    kind="scalar",
                    losses=frozenset(),
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic"}),
                ),
            }
        )

    def environment(self):
        return capture()

    def solve(self, problem, method, start, config):
        if method == "lm":
            optimum = problem.optima[0]
            status = RunStatus.CONVERGED
            x_final = optimum.x_star.tolist()
            cost_final = optimum.cost_star
            dist_to_opt = cost_gap = grad_norm_final = 0.0
            n_iter = n_feval = n_jeval = 1
            n_heval = 0
        else:
            status = RunStatus.ERROR
            x_final = cost_final = dist_to_opt = cost_gap = grad_norm_final = None
            n_iter = n_feval = n_jeval = n_heval = None
        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=x_final,
            cost_final=cost_final,
            dist_to_opt=dist_to_opt,
            cost_gap=cost_gap,
            grad_norm_final=grad_norm_final,
            status=status,
            n_iter=n_iter,
            n_feval=n_feval,
            n_jeval=n_jeval,
            n_heval=n_heval,
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp="2026-01-01T00:00:00Z",
        )


_EXPECTED_TABLES = [
    "baseline.csv",
    "baseline_basin_rates.csv",
    "large_residual.csv",
    "large_residual_basin_rates.csv",
    "ill_conditioning.csv",
    "robust_loss.csv",
    "bounded.csv",
    "scaling.csv",
    "dimensionality.csv",
    "derivative_source.csv",
    "scalar_cost.csv",
    "capability_matrix.csv",
    "typical.csv",
    "capability_frontier.csv",
]
_EXPECTED_PLOTS = [
    "large_residual.png",
    "ill_conditioning.png",
    "dimensionality.png",
    "scaling.png",
    "robust_loss.png",
    "capability_matrix.png",
    "typical.png",
    "capability_frontier.png",
]


def test_help_is_available_and_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])

    assert excinfo.value.code == 0


def test_report_help_is_available_and_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["report", "--help"])

    assert excinfo.value.code == 0


def test_report_command_defaults_to_the_reports_directory():
    args = build_parser().parse_args(["report"])

    assert args.output_dir == "reports"


def test_report_command_accepts_a_custom_output_dir():
    args = build_parser().parse_args(["report", "--output-dir", "custom"])

    assert args.output_dir == "custom"


def test_report_command_can_select_only_specific_studies():
    args = build_parser().parse_args(["report", "--only", "baseline", "scaling"])

    assert args.only == ["baseline", "scaling"]


def test_report_command_can_skip_specific_studies():
    args = build_parser().parse_args(["report", "--skip", "dimensionality"])

    assert args.skip == ["dimensionality"]


def test_report_command_can_skip_slow_studies():
    args = build_parser().parse_args(["report", "--skip-slow"])

    assert args.skip_slow is True
    assert build_parser().parse_args(["report"]).skip_slow is False


def test_only_and_skip_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--only", "baseline", "--skip", "scaling"])


def test_unknown_study_name_is_rejected_by_argument_parsing():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--only", "not-a-real-study"])


def test_report_command_can_select_specific_backends():
    args = build_parser().parse_args(["report", "--backends", "scipy"])

    assert args.backends == ["scipy"]


def test_report_command_backends_defaults_to_none():
    assert build_parser().parse_args(["report"]).backends is None


def test_unknown_backend_name_is_rejected_by_argument_parsing():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--backends", "not-a-real-backend"])


def test_select_studies_defaults_to_every_registered_study_except_parity_scatter():
    # parity_scatter is a two-backend comparison; the default n_backends=1
    # (matching a plain `trust-bench report`'s scipy-only default) excludes
    # it automatically, per MULTI_BACKEND_STUDIES.
    assert _select_studies() == set(STUDIES) - MULTI_BACKEND_STUDIES


def test_select_studies_includes_parity_scatter_when_two_backends_are_selected():
    assert _select_studies(n_backends=2) == set(STUDIES)


def test_select_studies_only_still_selects_parity_scatter_with_one_backend():
    # An explicit request is not silently dropped, unlike the default
    # "run everything" selection - the resulting ValueError from
    # parity_frame's own backend-count check is the appropriate response
    # to an impossible explicit request.
    assert _select_studies(only=["parity_scatter"]) == {"parity_scatter"}


def test_select_studies_only_returns_exactly_the_requested_set():
    assert _select_studies(only=["baseline", "scaling"]) == {"baseline", "scaling"}


def test_select_studies_skip_removes_from_the_full_set():
    assert _select_studies(skip=["dimensionality"]) == set(STUDIES) - {"dimensionality"} - MULTI_BACKEND_STUDIES


def test_select_studies_rejects_an_unknown_name_when_called_directly():
    # argparse's choices= already rejects an unknown name at the CLI
    # boundary (test_unknown_study_name_is_rejected_by_argument_parsing
    # above); this is the same guard for direct, non-CLI callers of
    # _select_studies/run_report.
    with pytest.raises(ValueError):
        _select_studies(only=["not-a-real-study"])


def test_select_studies_skip_slow_removes_the_slow_set():
    assert _select_studies(skip_slow=True, n_backends=2) == set(STUDIES) - SLOW_STUDIES


def test_select_backends_defaults_to_the_production_backends_list():
    assert _select_backends() == BACKENDS


def test_select_backends_only_returns_the_requested_backends():
    assert [b.name for b in _select_backends(["scipy"])] == ["scipy"]


def test_select_backends_rejects_an_unknown_name_when_called_directly():
    # argparse's choices= already rejects an unknown name at the CLI
    # boundary (test_unknown_backend_name_is_rejected_by_argument_parsing
    # above); this is the same guard for direct, non-CLI callers of
    # _select_backends/run_report.
    with pytest.raises(ValueError):
        _select_backends(["not-a-real-backend"])


def test_available_backends_is_keyed_by_each_backends_own_name():
    for name, backend in AVAILABLE_BACKENDS.items():
        assert backend.name == name


def test_run_report_writes_only_the_selected_studys_artefact(tmp_path):
    run_report(tmp_path, only=["baseline"])

    assert (tmp_path / "baseline.csv").exists()
    assert not (tmp_path / "scaling.csv").exists()


def test_run_report_uses_the_default_backend_when_none_selected(tmp_path):
    run_report(tmp_path, only=["baseline"])

    df = pd.read_csv(tmp_path / "baseline.csv")
    assert set(df["backend"]) == {"scipy"}


def test_run_report_writes_baselines_basin_of_attraction_rate_table(tmp_path):
    run_report(tmp_path, only=["baseline"])

    df = pd.read_csv(tmp_path / "baseline_basin_rates.csv")
    assert set(df.columns) >= {"problem_id", "backend", "basin_rate"}
    assert len(df) == len(CANONICAL_PROBLEMS)
    assert df["basin_rate"].between(0.0, 1.0).all()


def test_run_report_writes_large_residuals_basin_of_attraction_rate_table(tmp_path):
    run_report(tmp_path, only=["large_residual"])

    df = pd.read_csv(tmp_path / "large_residual_basin_rates.csv")
    assert set(df.columns) >= {"rho", "backend", "basin_rate"}
    assert len(df) == len(LARGE_RESIDUAL_RHOS)
    assert df["basin_rate"].between(0.0, 1.0).all()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_run_report_uses_the_selected_backends(tmp_path):
    run_report(tmp_path, only=["baseline"], backends=["trust-apl"])

    df = pd.read_csv(tmp_path / "baseline.csv")
    assert set(df["backend"]) == {"trust-apl"}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_run_report_writes_the_parity_scatter_only_with_two_backends(tmp_path):
    run_report(tmp_path, only=["parity_scatter"], backends=["scipy", "trust-apl"])

    df = pd.read_csv(tmp_path / "parity_scatter.csv")
    assert {"dist_to_opt_scipy", "dist_to_opt_trust-apl", "converged", "study"} <= set(df.columns)
    assert len(df) > 0
    assert (tmp_path / "parity_scatter.png").exists()


def test_run_report_raises_clearly_when_parity_scatter_is_explicitly_selected_with_one_backend(tmp_path):
    with pytest.raises(ValueError, match="two backends"):
        run_report(tmp_path, only=["parity_scatter"])


def test_run_report_measures_real_timing_for_every_study_it_runs(tmp_path, monkeypatch):
    # A study's own sweep() never opts into RunConfig.measure_timing
    # itself (it would slow down every test that calls the same
    # function directly for correctness, not performance) - run_report
    # is the one real invocation path that must actually produce
    # TimingStats, per this issue's own acceptance criterion.
    from trust_bench.core import runner as runner_module

    observed = {}

    def spy_study(output_dir, backends):
        observed["measuring"] = runner_module._measure_timing.get()

    monkeypatch.setitem(STUDIES, "baseline", spy_study)

    run_report(tmp_path, only=["baseline"])

    assert observed["measuring"] is True


def test_run_report_raises_clearly_when_a_study_does_not_support_the_selected_backend(tmp_path, monkeypatch):
    stub = _AlwaysErrorsBackend()
    monkeypatch.setitem(AVAILABLE_BACKENDS, stub.name, stub)

    with pytest.raises(ValueError, match=stub.name):
        run_report(tmp_path, only=["baseline"], backends=[stub.name])


def test_run_report_skips_a_study_with_a_known_backend_coverage_gap_instead_of_crashing(tmp_path, monkeypatch):
    # trust-apl has no evaluator for scalar_cost's Jacobian-free scalar
    # objectives at all: a known, permanent gap, not a bug - the report
    # must still produce every other artefact instead of aborting
    # entirely (the exact crash `trust-bench report --backends trust-apl
    # scipy` hits today).
    stub = _LmOnlyBackend()
    monkeypatch.setitem(AVAILABLE_BACKENDS, stub.name, stub)

    output_dir, skipped = run_report(tmp_path, only=["baseline", "scalar_cost"], backends=[stub.name])

    assert (output_dir / "baseline.csv").exists()
    assert not (output_dir / "scalar_cost.csv").exists()
    assert "scalar_cost" in skipped
    assert stub.name in skipped["scalar_cost"]


def test_run_report_does_not_absorb_an_unrelated_value_error_as_a_coverage_gap(tmp_path, monkeypatch):
    # Only _check_backend_coverage's own CoverageError is a known,
    # expected coverage gap; a plain ValueError from anywhere else in a
    # study's write path is a real bug and must still crash the report,
    # not get silently recorded in `skipped` alongside genuine gaps. A
    # second, healthy study is selected alongside the broken one so a
    # too-broad catch (absorb-and-continue, only raising once every
    # single selected study has failed) can't masquerade as correct
    # propagation the way it would with only one study selected.
    def _broken_writer(output_dir, backends):
        raise ValueError("unrelated bug: malformed input, not a coverage gap")

    monkeypatch.setitem(STUDIES, "baseline", _broken_writer)

    with pytest.raises(ValueError, match="unrelated bug"):
        run_report(tmp_path, only=["baseline", "large_residual"])


def test_main_prints_a_note_for_each_skipped_study(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "trust_bench.cli.run_report",
        lambda *args, **kwargs: (tmp_path, {"scalar_cost": "scalar_cost: no results for backend(s) trust-apl"}),
    )

    main(["report", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    # A properly unpacked, human-readable note - not main() naively
    # printing the raw (output_dir, skipped) tuple it now receives.
    assert "Skipped scalar_cost: scalar_cost: no results for backend(s) trust-apl" in captured.err
    assert "PosixPath" not in captured.out
    assert f"Report artefacts written to {tmp_path}" in captured.out


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_run_report_with_trust_apl_and_scipy_does_not_crash_on_scalar_costs_known_gap(tmp_path):
    # The direct reading of the reported bug: this exact invocation
    # currently raises and aborts before writing anything.
    output_dir, skipped = run_report(
        tmp_path, backends=["trust-apl", "scipy"], skip_slow=True, skip=["parity_scatter"]
    )

    assert "scalar_cost" in skipped
    assert (output_dir / "baseline.csv").exists()
    assert (output_dir / "typical.csv").exists()


def test_report_command_html_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).html is False


def test_report_command_can_request_an_html_bundle():
    assert build_parser().parse_args(["report", "--html"]).html is True


def test_run_report_does_not_write_an_html_bundle_by_default(tmp_path):
    run_report(tmp_path, only=["baseline"])

    assert not (tmp_path / "report.html").exists()


def test_run_report_writes_an_html_bundle_when_requested(tmp_path):
    run_report(tmp_path, only=["baseline"], html=True)

    html = (tmp_path / "report.html").read_text()
    assert "Baseline correctness" in html


@pytest.mark.slow
def test_report_command_produces_every_milestone_artefact(tmp_path):
    # The direct reading of the acceptance criterion: the full
    # Python-only pipeline, run end-to-end through the actual CLI
    # entry point, headless, producing real files on disk.
    main(["report", "--output-dir", str(tmp_path), "--html"])

    assert matplotlib.get_backend().lower() == "agg"

    for name in _EXPECTED_TABLES:
        path = tmp_path / name
        assert path.exists(), name
        df = pd.read_csv(path)
        assert len(df) > 0, name

    for name in _EXPECTED_PLOTS:
        path = tmp_path / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    assert (tmp_path / "report.html").exists()
