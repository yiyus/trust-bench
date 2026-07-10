import matplotlib
import pandas as pd
import pytest

from trust_bench.cli import SLOW_STUDIES, STUDIES, _select_studies, build_parser, main, run_report

_EXPECTED_TABLES = [
    "baseline.csv",
    "large_residual.csv",
    "ill_conditioning.csv",
    "robust_loss.csv",
    "bounded.csv",
    "scaling.csv",
    "dimensionality.csv",
    "derivative_source.csv",
    "capability_matrix.csv",
]
_EXPECTED_PLOTS = ["large_residual.png"]


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


def test_run_report_writes_only_the_selected_studys_artefact(tmp_path):
    run_report(tmp_path, only=["baseline"])

    assert (tmp_path / "baseline.csv").exists()
    assert not (tmp_path / "scaling.csv").exists()


@pytest.mark.slow
def test_report_command_produces_every_milestone_artefact(tmp_path):
    # The direct reading of the acceptance criterion: the full
    # Python-only pipeline, run end-to-end through the actual CLI
    # entry point, headless, producing real files on disk.
    main(["report", "--output-dir", str(tmp_path)])

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
