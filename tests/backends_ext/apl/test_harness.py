import json
import shutil
import subprocess
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[3] / "backends_ext" / "apl" / "run_harness.sh"

pytestmark = pytest.mark.skipif(shutil.which("dyalogscript") is None, reason="Dyalog APL is not installed")


def _run(request, tmp_path, timeout=60):
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(request))
    proc = subprocess.run(
        ["bash", str(_HARNESS), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc, json.loads(output_path.read_text())


@pytest.mark.slow
def test_solves_rosenbrock_from_the_standard_start_to_convergence(tmp_path):
    proc, result = _run({"problem_id": "rosenbrock", "x0": [-1.2, 1.0]}, tmp_path)

    assert proc.returncode == 0
    assert result["problem_id"] == "rosenbrock"
    assert result["status"] == "CONVERGED"
    assert result["message"] is None
    assert result["x_final"] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert result["cost_final"] == pytest.approx(0.0, abs=1e-9)
    assert result["grad_norm_final"] < 1e-4
    assert result["n_iter"] > 0
    assert result["n_feval"] > result["n_iter"]
    assert result["n_jeval"] is None
    assert result["n_heval"] is None


@pytest.mark.slow
def test_honours_an_explicit_tolerance_and_loss_and_max_iter(tmp_path):
    request = {
        "problem_id": "rosenbrock",
        "x0": [-1.2, 1.0],
        "max_iter": 1000,
        "tolerance": 1e-10,
        "loss": "L2",
    }
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"] == pytest.approx([1.0, 1.0], abs=1e-6)


@pytest.mark.slow
def test_reports_max_iter_status_when_the_iteration_cap_is_reached(tmp_path):
    proc, result = _run({"problem_id": "rosenbrock", "x0": [-1.2, 1.0], "max_iter": 1}, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "MAX_ITER"
    assert result["message"] is None
    assert result["x_final"] is not None
    assert result["n_iter"] > 0


@pytest.mark.slow
def test_reports_error_status_for_an_unknown_problem_id(tmp_path):
    proc, result = _run({"problem_id": "no-such-problem", "x0": [0.0, 0.0]}, tmp_path)

    assert proc.returncode == 1
    assert result["status"] == "ERROR"
    assert "no-such-problem" in result["message"]
    assert result["x_final"] is None
    assert result["cost_final"] is None


@pytest.mark.slow
def test_reports_error_status_for_malformed_input(tmp_path):
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text("not valid json{{{")

    proc = subprocess.run(
        ["bash", str(_HARNESS), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(output_path.read_text())

    assert proc.returncode == 1
    assert result["status"] == "ERROR"
    assert result["message"] is not None
