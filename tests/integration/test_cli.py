import shutil
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from trust_bench.cli import (
    AVAILABLE_BACKENDS,
    MULTI_BACKEND_STUDIES,
    SLOW_STUDIES,
    STUDIES,
    _select_backends,
    _select_studies,
    build_parser,
    main,
    run_compare,
    run_report,
)
from trust_bench.core import storage
from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import EnvProvenance, capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.core.storage import load
from trust_bench.problems import CANONICAL_PROBLEMS
from trust_bench.reporting.html_report import TITLES, build_html_report
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


_FIXTURE_REPORT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "report"


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


def test_only_and_skip_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--only", "baseline", "--skip", "scaling"])


def test_unknown_study_name_is_rejected_by_argument_parsing():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--only", "not-a-real-study"])


def test_report_command_full_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).full is False


def test_report_command_can_request_the_full_report():
    assert build_parser().parse_args(["report", "--full"]).full is True


def test_report_command_scipy_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).scipy is False


def test_report_command_can_request_scipy():
    assert build_parser().parse_args(["report", "--scipy"]).scipy is True


def test_report_command_no_trust_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).no_trust is False


def test_report_command_can_disable_trust():
    assert build_parser().parse_args(["report", "--no-trust"]).no_trust is True


def test_report_command_baselines_default_to_an_empty_list():
    assert build_parser().parse_args(["report"]).baselines == []


def test_report_command_accepts_baseline_directories():
    args = build_parser().parse_args(["report", "old-reports", "other-reports"])

    assert args.baselines == ["old-reports", "other-reports"]


def test_select_studies_defaults_to_every_registered_study_except_slow_and_multi_backend_ones():
    # parity_scatter needs two backends (excluded by the default single
    # backend, per MULTI_BACKEND_STUDIES); the slow studies are skipped
    # by default now that `report` is expected to run frequently, feeding
    # the longitudinal ledger - --full opts back into them.
    assert _select_studies() == set(STUDIES) - MULTI_BACKEND_STUDIES - SLOW_STUDIES


def test_select_studies_includes_parity_scatter_when_two_backends_are_selected():
    assert _select_studies(skip_slow=False, n_backends=2) == set(STUDIES)


def test_select_studies_only_still_selects_parity_scatter_with_one_backend():
    # An explicit request is not silently dropped, unlike the default
    # "run everything" selection - the resulting ValueError from
    # parity_frame's own backend-count check is the appropriate response
    # to an impossible explicit request.
    assert _select_studies(only=["parity_scatter"]) == {"parity_scatter"}


def test_select_studies_only_still_selects_a_slow_study_explicitly():
    # Same carve-out as parity_scatter above, for the default skip-slow
    # behaviour: naming a slow study via --only still runs it.
    assert _select_studies(only=["dimensionality"]) == {"dimensionality"}


def test_select_studies_only_returns_exactly_the_requested_set():
    assert _select_studies(only=["baseline", "scaling"]) == {"baseline", "scaling"}


def test_select_studies_skip_removes_from_the_default_set():
    assert (
        _select_studies(skip=["large_residual"])
        == set(STUDIES) - {"large_residual"} - MULTI_BACKEND_STUDIES - SLOW_STUDIES
    )


def test_select_studies_rejects_an_unknown_name_when_called_directly():
    # argparse's choices= already rejects an unknown name at the CLI
    # boundary (test_unknown_study_name_is_rejected_by_argument_parsing
    # above); this is the same guard for direct, non-CLI callers of
    # _select_studies/run_report.
    with pytest.raises(ValueError):
        _select_studies(only=["not-a-real-study"])


def test_select_studies_full_report_includes_the_slow_set():
    assert _select_studies(skip_slow=False, n_backends=2) == set(STUDIES)


def test_select_backends_defaults_to_trust_apl_only():
    assert [b.name for b in _select_backends()] == ["trust-apl"]


def test_select_backends_can_add_scipy():
    assert {b.name for b in _select_backends(use_scipy=True)} == {"trust-apl", "scipy"}


def test_select_backends_can_disable_trust():
    assert [b.name for b in _select_backends(use_trust=False, use_scipy=True)] == ["scipy"]


def test_select_backends_raises_when_nothing_is_selected():
    with pytest.raises(ValueError, match="no backend"):
        _select_backends(use_trust=False, use_scipy=False)


def test_available_backends_is_keyed_by_each_backends_own_name():
    for name, backend in AVAILABLE_BACKENDS.items():
        assert backend.name == name


def test_run_report_writes_only_the_selected_studys_artefact(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    assert (tmp_path / "baseline.csv").exists()
    assert not (tmp_path / "scaling.csv").exists()


def test_run_report_can_select_scipy_only(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    df = pd.read_csv(tmp_path / "baseline.csv")
    assert set(df["backend"]) == {"scipy"}


def test_run_report_writes_baselines_basin_of_attraction_rate_table(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    df = pd.read_csv(tmp_path / "baseline_basin_rates.csv")
    assert set(df.columns) >= {"problem_id", "backend", "basin_rate"}
    assert len(df) == len(CANONICAL_PROBLEMS)
    assert df["basin_rate"].between(0.0, 1.0).all()


def test_run_report_writes_large_residuals_basin_of_attraction_rate_table(tmp_path):
    run_report(tmp_path, only=["large_residual"], use_trust=False, use_scipy=True, html=False)

    df = pd.read_csv(tmp_path / "large_residual_basin_rates.csv")
    assert set(df.columns) >= {"rho", "backend", "basin_rate"}
    assert len(df) == len(LARGE_RESIDUAL_RHOS)
    assert df["basin_rate"].between(0.0, 1.0).all()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_run_report_uses_trust_apl_by_default(tmp_path):
    run_report(tmp_path, only=["baseline"], html=False)

    df = pd.read_csv(tmp_path / "baseline.csv")
    assert set(df["backend"]) == {"trust-apl"}


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")
def test_run_report_writes_the_parity_scatter_only_with_two_backends(tmp_path):
    run_report(tmp_path, only=["parity_scatter"], use_scipy=True, html=False)

    df = pd.read_csv(tmp_path / "parity_scatter.csv")
    assert {"dist_to_opt_scipy", "dist_to_opt_trust-apl", "converged", "study"} <= set(df.columns)
    assert len(df) > 0
    assert (tmp_path / "parity_scatter.png").exists()


def test_run_report_raises_clearly_when_parity_scatter_is_explicitly_selected_with_one_backend(tmp_path):
    with pytest.raises(ValueError, match="two backends"):
        run_report(tmp_path, only=["parity_scatter"], html=False)


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

    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    assert observed["measuring"] is True


def test_run_report_raises_clearly_when_a_study_does_not_support_the_selected_backend(tmp_path, monkeypatch):
    stub = _AlwaysErrorsBackend()
    monkeypatch.setitem(AVAILABLE_BACKENDS, "scipy", stub)

    with pytest.raises(ValueError, match=stub.name):
        run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)


def test_run_report_skips_a_study_with_a_known_backend_coverage_gap_instead_of_crashing(tmp_path, monkeypatch):
    # trust-apl has no evaluator for scalar_cost's Jacobian-free scalar
    # objectives at all: a known, permanent gap, not a bug - the report
    # must still produce every other artefact instead of aborting
    # entirely.
    stub = _LmOnlyBackend()
    monkeypatch.setitem(AVAILABLE_BACKENDS, "scipy", stub)

    output_dir, skipped = run_report(
        tmp_path, only=["baseline", "scalar_cost"], use_trust=False, use_scipy=True, html=False
    )

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
        run_report(tmp_path, only=["baseline", "large_residual"], use_trust=False, use_scipy=True)


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
    # currently raises and aborts before writing anything. Only the
    # affected study plus one unaffected study are selected - the bug
    # is specific to scalar_cost's own coverage gap, not to running the
    # rest of the report alongside it.
    output_dir, skipped = run_report(
        tmp_path, only=["scalar_cost", "baseline"], use_scipy=True, html=False
    )

    assert "scalar_cost" in skipped
    assert (output_dir / "baseline.csv").exists()


def test_report_command_no_html_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).no_html is False


def test_report_command_can_request_no_html():
    assert build_parser().parse_args(["report", "--no-html"]).no_html is True


def test_run_report_writes_an_html_bundle_by_default(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True)

    html = (tmp_path / "report.html").read_text()
    assert "Baseline correctness" in html


def test_run_report_can_disable_the_html_bundle(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    assert not (tmp_path / "report.html").exists()


def test_report_command_no_results_flag_defaults_to_false():
    assert build_parser().parse_args(["report"]).no_results is False


def test_report_command_can_request_no_results():
    assert build_parser().parse_args(["report", "--no-results"]).no_results is True


def test_run_report_writes_a_results_jsonl_file_by_default(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    jsonl_files = list((tmp_path / "results").glob("*.jsonl"))
    assert len(jsonl_files) == 1
    df = load(jsonl_files[0])
    assert len(df) > 0
    assert set(df["problem_id"]) & {problem.id for problem in CANONICAL_PROBLEMS}


def test_run_report_names_the_results_file_by_the_harness_git_sha(tmp_path, monkeypatch):
    monkeypatch.setattr("trust_bench.cli.harness_git_sha", lambda: "fixed-sha")

    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    assert (tmp_path / "results" / "fixed-sha.jsonl").exists()


def test_run_report_appends_across_repeated_calls_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr("trust_bench.cli.harness_git_sha", lambda: "fixed-sha")
    path = tmp_path / "results" / "fixed-sha.jsonl"

    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)
    first_len = len(load(path))
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)
    second_len = len(load(path))

    assert second_len == first_len * 2


def test_run_report_can_disable_results_persistence(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False, results_dir=None)

    assert not (tmp_path / "results").exists()


def test_main_report_writes_results_by_default(tmp_path):
    main(["report", "--output-dir", str(tmp_path), "--only", "baseline", "--no-trust", "--scipy"])

    assert (tmp_path / "results").exists()


def test_main_report_disables_results_persistence_when_no_results_flag_is_passed(tmp_path):
    main(
        [
            "report",
            "--output-dir",
            str(tmp_path),
            "--only",
            "baseline",
            "--no-trust",
            "--scipy",
            "--no-results",
        ]
    )

    assert not (tmp_path / "results").exists()


def test_main_report_wires_every_new_flag_through_to_run_report(monkeypatch, tmp_path):
    captured = {}

    def spy_run_report(output_dir, **kwargs):
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        return output_dir, {}

    monkeypatch.setattr("trust_bench.cli.run_report", spy_run_report)

    main(
        [
            "report",
            "some-baseline",
            "--output-dir",
            str(tmp_path),
            "--full",
            "--scipy",
            "--no-trust",
            "--no-html",
            "--no-results",
        ]
    )

    assert captured["skip_slow"] is False
    assert captured["use_scipy"] is True
    assert captured["use_trust"] is False
    assert captured["html"] is False
    assert captured["results_dir"] is None
    assert captured["baselines"] == ["some-baseline"]


def test_run_report_writes_a_comparison_against_a_given_baseline_directory(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_report(baseline_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    run_report(
        candidate_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False, baselines=[baseline_dir]
    )

    assert (candidate_dir / "compare.csv").exists()
    assert (candidate_dir / "compare.png").exists()


def test_run_report_folds_the_comparison_into_the_html_report_when_a_baseline_is_given(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_report(baseline_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    run_report(candidate_dir, only=["baseline"], use_trust=False, use_scipy=True, baselines=[baseline_dir])

    html = (candidate_dir / "report.html").read_text()
    assert "Longitudinal comparison" in html


# baseline's own registered problems don't all share the same number of
# starts (rosenbrock alone has "standard" and "far") - one comparison row
# per distinct (problem_id, start) pair, not one per problem.
_BASELINE_STANDARD_STARTS = sum(len(problem.starts) for problem in CANONICAL_PROBLEMS)


def test_run_report_pools_multiple_baseline_directories(tmp_path):
    baseline_a = tmp_path / "baseline_a"
    baseline_b = tmp_path / "baseline_b"
    candidate_dir = tmp_path / "candidate"
    run_report(baseline_a, only=["baseline"], use_trust=False, use_scipy=True, html=False)
    run_report(baseline_b, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    run_report(
        candidate_dir,
        only=["baseline"],
        use_trust=False,
        use_scipy=True,
        html=False,
        baselines=[baseline_a, baseline_b],
    )

    df = pd.read_csv(candidate_dir / "compare.csv")
    assert len(df) == _BASELINE_STANDARD_STARTS


def test_run_report_writes_exactly_one_comparison_row_per_key_despite_baselines_own_duplicate_start(tmp_path):
    # baseline.py's basin_rates() re-probes the "standard" start already
    # covered by standard_start_results() - both go through run(), so a
    # single `only=["baseline"]` run_report call genuinely writes two
    # rows for that key to results/*.jsonl. compare must still produce
    # one row per (problem_id, backend, method, start), not a cross-join
    # explosion.
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_report(baseline_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    run_report(
        candidate_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False, baselines=[baseline_dir]
    )

    df = pd.read_csv(candidate_dir / "compare.csv")
    assert len(df) == _BASELINE_STANDARD_STARTS
    assert df["classification"].notna().all()


def test_run_report_can_compare_against_a_baseline_with_a_custom_results_dir(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_report(baseline_dir, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    run_report(
        candidate_dir,
        only=["baseline"],
        use_trust=False,
        use_scipy=True,
        html=False,
        results_dir="custom_results",
        baselines=[baseline_dir],
    )

    assert (candidate_dir / "compare.csv").exists()
    assert (candidate_dir / "custom_results").exists()
    assert not (candidate_dir / "results").exists()


def test_run_report_without_a_baseline_writes_no_comparison_artefacts(tmp_path):
    run_report(tmp_path, only=["baseline"], use_trust=False, use_scipy=True, html=False)

    assert not (tmp_path / "compare.csv").exists()


def test_run_report_raises_clearly_when_comparing_with_results_persistence_disabled(tmp_path):
    with pytest.raises(ValueError, match="results"):
        run_report(
            tmp_path,
            only=["baseline"],
            use_trust=False,
            use_scipy=True,
            html=False,
            results_dir=None,
            baselines=[tmp_path / "nonexistent"],
        )


def test_report_command_accepts_baseline_directories_end_to_end(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    main(["report", "--output-dir", str(baseline_dir), "--only", "baseline", "--no-trust", "--scipy", "--no-html"])

    main(
        [
            "report",
            str(baseline_dir),
            "--output-dir",
            str(candidate_dir),
            "--only",
            "baseline",
            "--no-trust",
            "--scipy",
            "--no-html",
        ]
    )

    assert (candidate_dir / "compare.csv").exists()


def test_report_command_runs_end_to_end_through_the_real_cli_entry_point(tmp_path):
    # A cheap, real reading of the acceptance criterion: the actual CLI
    # entry point, headless, writing real files on disk for a small,
    # fast subset - proving the orchestration wiring (STUDIES loop, the
    # html-by-default bundling) works end-to-end without paying for
    # every study (dimensionality and capability_frontier alone cost
    # tens of seconds each by design; see SLOW_STUDIES) or requiring
    # dyalogscript (--no-trust --scipy).
    main(["report", "--output-dir", str(tmp_path), "--only", "baseline", "typical", "--no-trust", "--scipy"])

    assert matplotlib.get_backend().lower() == "agg"

    for name in ["baseline.csv", "baseline_basin_rates.csv", "typical.csv", "typical.png"]:
        path = tmp_path / name
        assert path.exists(), name

    assert (tmp_path / "report.html").exists()


def _compare_provenance(**overrides):
    defaults = dict(
        backend_name="scipy",
        backend_version="1.11.0",
        language_runtime="CPython 3.11.0",
        blas_lapack="openblas",
        os="Linux 6.0",
        cpu_model="generic",
        cpu_count=4,
        machine_fingerprint="fp-a",
    )
    defaults.update(overrides)
    return EnvProvenance(**defaults)


def _compare_run_result(**overrides):
    defaults = dict(
        problem_id="rosenbrock",
        backend="scipy",
        method="lm",
        start="standard",
        x_final=[1.0, 1.0],
        cost_final=0.0,
        dist_to_opt=0.0,
        cost_gap=0.0,
        grad_norm_final=0.0,
        status=RunStatus.CONVERGED,
        n_iter=5,
        n_feval=10,
        n_jeval=5,
        n_heval=0,
        trace=None,
        timing=None,
        config={"ftol": 1e-8},
        provenance=_compare_provenance(),
        harness_git_sha="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def _write_results_dir(directory, results, sha="sha-a"):
    directory = Path(directory)
    path = directory / "results" / f"{sha}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        storage.append(result, path)
    return directory


def test_compare_help_is_available_and_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["compare", "--help"])

    assert excinfo.value.code == 0


def test_compare_command_parses_baseline_and_candidate_directories():
    args = build_parser().parse_args(["compare", "baseline-dir", "candidate-dir"])

    assert args.baseline_dir == "baseline-dir"
    assert args.candidate_dir == "candidate-dir"
    assert args.output_dir == "reports"
    assert args.html is False


def test_compare_command_accepts_output_dir_and_html_flag():
    args = build_parser().parse_args(
        ["compare", "baseline-dir", "candidate-dir", "--output-dir", "out", "--html"]
    )

    assert args.output_dir == "out"
    assert args.html is True


def test_run_compare_writes_a_full_provenance_table_with_classifications(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result(dist_to_opt=0.0)])
    candidate_dir = _write_results_dir(tmp_path / "candidate", [_compare_run_result(dist_to_opt=0.5)])

    output_dir = run_compare(baseline_dir, candidate_dir, tmp_path / "out")

    df = pd.read_csv(output_dir / "compare.csv")
    assert df["classification"].iloc[0] == "regression"
    assert "baseline_machine_fingerprint" in df.columns
    assert "candidate_machine_fingerprint" in df.columns


def test_run_compare_writes_a_classification_counts_plot(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result()])
    candidate_dir = _write_results_dir(tmp_path / "candidate", [_compare_run_result()])

    output_dir = run_compare(baseline_dir, candidate_dir, tmp_path / "out")

    assert (output_dir / "compare.png").exists()


def test_run_compare_does_not_write_an_html_bundle_by_default(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result()])
    candidate_dir = _write_results_dir(tmp_path / "candidate", [_compare_run_result()])

    output_dir = run_compare(baseline_dir, candidate_dir, tmp_path / "out")

    assert not (output_dir / "report.html").exists()


def test_run_compare_writes_an_html_bundle_when_requested(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result(dist_to_opt=0.0)])
    candidate_dir = _write_results_dir(tmp_path / "candidate", [_compare_run_result(dist_to_opt=0.5)])

    output_dir = run_compare(baseline_dir, candidate_dir, tmp_path / "out", html=True)

    html = (output_dir / "report.html").read_text()
    assert "regression" in html


def test_run_compare_pools_multiple_jsonl_files_within_a_directory(tmp_path):
    baseline_dir = tmp_path / "baseline"
    _write_results_dir(baseline_dir, [_compare_run_result(problem_id="rosenbrock")], sha="sha-a")
    _write_results_dir(baseline_dir, [_compare_run_result(problem_id="beale")], sha="sha-b")
    candidate_dir = _write_results_dir(
        tmp_path / "candidate",
        [_compare_run_result(problem_id="rosenbrock"), _compare_run_result(problem_id="beale")],
    )

    output_dir = run_compare(baseline_dir, candidate_dir, tmp_path / "out")

    df = pd.read_csv(output_dir / "compare.csv")
    assert len(df) == 2


def test_run_compare_raises_clearly_when_a_directory_has_no_results(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result()])
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="no results"):
        run_compare(baseline_dir, empty_dir, tmp_path / "out")


def test_compare_command_runs_end_to_end_through_the_real_cli_entry_point(tmp_path):
    baseline_dir = _write_results_dir(tmp_path / "baseline", [_compare_run_result()])
    candidate_dir = _write_results_dir(tmp_path / "candidate", [_compare_run_result()])

    main(
        [
            "compare",
            str(baseline_dir),
            str(candidate_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--html",
        ]
    )

    assert (tmp_path / "out" / "compare.csv").exists()
    assert (tmp_path / "out" / "report.html").exists()


def test_build_html_report_renders_every_milestone_artefact_from_a_committed_fixture_set():
    # tests/fixtures/report/ is a real report's output, committed once
    # so this test exercises build_html_report's full assembly logic -
    # every tier, every theme, every study's own table/plot/caption -
    # without ever running a solve. scalar_cost is correctly absent:
    # the fixture was generated with two backends, and scalar_cost has
    # a known, permanent trust-apl coverage gap.
    html = build_html_report(_FIXTURE_REPORT_DIR)

    for name, title in TITLES.items():
        if (_FIXTURE_REPORT_DIR / f"{name}.csv").exists() or (_FIXTURE_REPORT_DIR / f"{name}.png").exists():
            assert title in html, name
