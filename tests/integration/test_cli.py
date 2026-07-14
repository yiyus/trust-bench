import shutil

import matplotlib
import pandas as pd
import pytest

from trust_bench.backends import BACKENDS
from trust_bench.cli import (
    AVAILABLE_BACKENDS,
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
]
_EXPECTED_PLOTS = [
    "large_residual.png",
    "ill_conditioning.png",
    "dimensionality.png",
    "scaling.png",
    "robust_loss.png",
    "capability_matrix.png",
    "typical.png",
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


def test_select_studies_defaults_to_every_registered_study():
    assert _select_studies() == set(STUDIES)


def test_select_studies_only_returns_exactly_the_requested_set():
    assert _select_studies(only=["baseline", "scaling"]) == {"baseline", "scaling"}


def test_select_studies_skip_removes_from_the_full_set():
    assert _select_studies(skip=["dimensionality"]) == set(STUDIES) - {"dimensionality"}


def test_select_studies_rejects_an_unknown_name_when_called_directly():
    # argparse's choices= already rejects an unknown name at the CLI
    # boundary (test_unknown_study_name_is_rejected_by_argument_parsing
    # above); this is the same guard for direct, non-CLI callers of
    # _select_studies/run_report.
    with pytest.raises(ValueError):
        _select_studies(only=["not-a-real-study"])


def test_select_studies_skip_slow_removes_the_slow_set():
    assert _select_studies(skip_slow=True) == set(STUDIES) - SLOW_STUDIES


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


def test_run_report_raises_clearly_when_a_study_does_not_support_the_selected_backend(tmp_path, monkeypatch):
    stub = _AlwaysErrorsBackend()
    monkeypatch.setitem(AVAILABLE_BACKENDS, stub.name, stub)

    with pytest.raises(ValueError, match=stub.name):
        run_report(tmp_path, only=["baseline"], backends=[stub.name])


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
    assert "baseline" in html


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
