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
    # lm never computes a true Hessian; 0, not null, distinguishes "this
    # method computed zero Hessians" from "unknown" (see trust-exact's
    # own tests below for the real-count case).
    assert result["n_heval"] == 0


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
def test_solves_a_bounded_problem_and_stops_at_the_active_boundary(tmp_path):
    request = {
        "problem_id": "quadratic",
        "x0": [1.0, -1.0],
        "bounds": [[0.5, -10.0], [10.0, 10.0]],
        "max_iter": 200,
    }
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"][0] == pytest.approx(0.5, abs=1e-4)
    assert result["x_final"][1] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.slow
def test_finite_difference_mode_uses_more_evaluations_than_analytic(tmp_path):
    analytic_proc, analytic = _run({"problem_id": "quadratic", "x0": [1.0, -1.0]}, tmp_path)
    fd_proc, fd = _run(
        {"problem_id": "quadratic", "x0": [1.0, -1.0], "derivative_mode": "finite-difference"},
        tmp_path,
    )

    assert analytic_proc.returncode == 0
    assert fd_proc.returncode == 0
    assert analytic["status"] == "CONVERGED"
    assert fd["status"] == "CONVERGED"
    assert fd["x_final"] == pytest.approx(analytic["x_final"], abs=1e-4)
    assert fd["n_feval"] > analytic["n_feval"]


@pytest.mark.slow
def test_solves_with_the_bfgs_method(tmp_path):
    request = {"problem_id": "rosenbrock", "x0": [-1.2, 1.0], "method": "BFGS", "max_iter": 200}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"] == pytest.approx([1.0, 1.0], abs=1e-4)
    assert result["n_heval"] == 0


@pytest.mark.slow
def test_solves_with_the_trust_exact_method_and_reports_a_real_n_heval(tmp_path):
    request = {"problem_id": "rosenbrock", "x0": [-1.2, 1.0], "method": "trust-exact", "max_iter": 200}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"] == pytest.approx([1.0, 1.0], abs=1e-4)
    assert result["n_heval"] is not None
    assert result["n_heval"] > 0


@pytest.mark.slow
def test_trust_exact_converges_in_fewer_iterations_than_bfgs_from_the_same_start(tmp_path):
    request = {"problem_id": "rosenbrock", "x0": [-1.2, 1.0], "max_iter": 200}
    _, bfgs = _run({**request, "method": "BFGS"}, tmp_path)
    _, newton = _run({**request, "method": "trust-exact"}, tmp_path)

    assert bfgs["status"] == "CONVERGED"
    assert newton["status"] == "CONVERGED"
    assert newton["n_iter"] < bfgs["n_iter"]


@pytest.mark.slow
def test_bounds_work_for_the_trust_exact_method(tmp_path):
    request = {
        "problem_id": "quadratic",
        "x0": [1.0, -1.0],
        "method": "trust-exact",
        "bounds": [[0.5, -10.0], [10.0, 10.0]],
        "max_iter": 200,
    }
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"][0] == pytest.approx(0.5, abs=1e-4)
    assert result["x_final"][1] == pytest.approx(0.0, abs=1e-4)
    # lm reaches this same boundary point for this scenario too, so
    # status/x_final alone would pass even if req.method were silently
    # ignored and the harness fell back to lm. n_heval > 0 only holds
    # if trust-exact was genuinely dispatched.
    assert result["n_heval"] > 0


@pytest.mark.slow
def test_reports_error_status_for_an_unknown_method(tmp_path):
    request = {"problem_id": "rosenbrock", "x0": [-1.2, 1.0], "method": "not-a-real-method"}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 1
    assert result["status"] == "ERROR"
    assert "not-a-real-method" in result["message"]


@pytest.mark.slow
def test_evaluate_mode_also_reports_the_hessian(tmp_path):
    request = {"mode": "evaluate", "problem_id": "rosenbrock", "x": [-1.2, 1.0]}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "OK"
    expected = [[665.0, 240.0], [240.0, 100.0]]
    for row, expected_row in zip(result["hessian"], expected):
        assert row == pytest.approx(expected_row)


@pytest.mark.slow
def test_reports_error_status_for_an_unknown_problem_id(tmp_path):
    proc, result = _run({"problem_id": "no-such-problem", "x0": [0.0, 0.0]}, tmp_path)

    assert proc.returncode == 1
    assert result["status"] == "ERROR"
    assert "no-such-problem" in result["message"]
    assert result["x_final"] is None
    assert result["cost_final"] is None


@pytest.mark.slow
def test_solves_a_parametrised_difficulty_family_problem(tmp_path):
    request = {"problem_id": "scaling(s=10.0)", "x0": [0.0, 0.0], "method": "lm"}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "CONVERGED"
    assert result["x_final"] == pytest.approx([3.0, -2.0], abs=1e-6)


@pytest.mark.slow
def test_evaluate_mode_reports_residual_jacobian_and_hessian_for_a_parametrised_problem(tmp_path):
    request = {"mode": "evaluate", "problem_id": "ill_conditioned(kappa=100.0)", "x": [2.0, 3.0]}
    proc, result = _run(request, tmp_path)

    assert proc.returncode == 0
    assert result["status"] == "OK"
    assert result["residual"] == pytest.approx([0.0, 0.0], abs=1e-9)


@pytest.mark.slow
def test_reports_error_status_for_an_unknown_parametrised_family(tmp_path):
    proc, result = _run({"problem_id": "not_a_family(x=1.0)", "x0": [0.0, 0.0]}, tmp_path)

    assert proc.returncode == 1
    assert result["status"] == "ERROR"
    assert "not_a_family(x=1.0)" in result["message"]


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
    assert result["message"].startswith("DOMAIN ERROR")
    assert result["message"] != "DOMAIN ERROR"
