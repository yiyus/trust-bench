import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_trust_bench_package_is_importable():
    import trust_bench

    assert trust_bench is not None


def test_make_test_target_invokes_pytest():
    result = subprocess.run(
        ["make", "-n", "test"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "pytest" in result.stdout


def test_make_lint_target_invokes_ruff():
    result = subprocess.run(
        ["make", "-n", "lint"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "ruff" in result.stdout


def test_make_coverage_target_invokes_coverage():
    result = subprocess.run(
        ["make", "-n", "coverage"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert "coverage" in result.stdout


def test_ci_workflow_triggers_on_push_and_pull_request():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert re.search(r"^\s*push\s*:", workflow, re.MULTILINE)
    assert re.search(r"^\s*pull_request\s*:", workflow, re.MULTILINE)


def test_ci_workflow_runs_test_lint_and_coverage_targets():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for target in ("test", "lint", "coverage"):
        assert f"make {target}" in workflow
