import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus

_HARNESS = Path(__file__).resolve().parents[2] / "backends_ext" / "apl" / "run_harness.sh"

_STATUS = {
    "CONVERGED": RunStatus.CONVERGED,
    "MAX_ITER": RunStatus.MAX_ITER,
    "FAILED": RunStatus.FAILED,
    "ERROR": RunStatus.ERROR,
}

# RunConfig's loss vocabulary mapped to trust's Loss namespace: the five
# scipy also has (see scipy_backend.py's _LEAST_SQUARES_LOSSES), plus
# trust's own redescending losses scipy has no equivalent for.
_LOSS_TO_TRUST = {
    "linear": "L2",
    "soft_l1": "SoftL1",
    "huber": "Huber",
    "cauchy": "Cauchy",
    "arctan": "Arctan",
    "tukey": "Tukey",
    "welsch": "Welsch",
    "fair": "Fair",
}


_TIMEOUT_SECONDS = 60

# Matches error_result.dyalog's ErrorResult field set exactly, so a
# subprocess timeout is indistinguishable, from the caller's side, from
# any other harness-reported ERROR (evaluate_problem's status check and
# solve()'s _STATUS lookup both handle it the same way).
def _timeout_result(message: str) -> dict:
    return {
        "problem_id": None,
        "status": "ERROR",
        "message": message,
        "x_final": None,
        "cost_final": None,
        "n_iter": None,
        "n_feval": None,
        "n_jeval": None,
        "n_heval": None,
        "grad_norm_final": None,
    }


def _run_harness(request: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "request.json"
        output_path = Path(tmp) / "result.json"
        input_path.write_text(json.dumps(request))
        try:
            subprocess.run(
                ["bash", str(_HARNESS), str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return _timeout_result(f"harness did not complete within {_TIMEOUT_SECONDS}s")
        return json.loads(output_path.read_text())


def evaluate_problem(problem_id: str, x) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Evaluate a problem's residual, Jacobian and Hessian at x via the APL
    harness's 'evaluate' mode. Used by the cross-language parity tests; not
    called by APLBackend.solve, which never needs a probe at an arbitrary
    point.
    """
    request = {"mode": "evaluate", "problem_id": problem_id, "x": np.asarray(x, dtype=float).tolist()}
    response = _run_harness(request)
    if response["status"] != "OK":
        raise RuntimeError(response["message"])
    return response["residual"], response["jacobian"], response["hessian"]


def _dyalog_version() -> str:
    try:
        result = subprocess.run(
            ["dyalog", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    match = re.search(r"Version\s+(\S+)", result.stdout)
    return match.group(1) if match else "unknown"


class APLBackend(Backend):
    name = "trust-apl"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            methods={
                "lm": MethodCapabilities(
                    kind="residuals",
                    losses=frozenset(_LOSS_TO_TRUST),
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic", "finite-difference"}),
                ),
                "BFGS": MethodCapabilities(
                    kind="scalar",
                    losses=frozenset({"linear"}),
                    bounds=True,
                    analytic_hessian=False,
                    derivative_modes=frozenset({"analytic", "finite-difference"}),
                ),
                "trust-exact": MethodCapabilities(
                    kind="scalar",
                    losses=frozenset({"linear"}),
                    bounds=True,
                    analytic_hessian=True,
                    derivative_modes=frozenset({"analytic"}),
                ),
            }
        )

    def environment(self):
        # Same machine-level facts as the Python backends' own provenance;
        # only the runtime/version fields describe Dyalog rather than CPython.
        machine = capture()
        version = _dyalog_version()
        machine.backend_name = self.name
        machine.backend_version = version
        machine.language_runtime = f"Dyalog APL {version}"
        return machine

    def solve(self, problem, method: str, start: str, config) -> RunResult:
        if method not in self.capabilities().methods:
            raise ValueError(f"{self.name} has no method {method!r}")
        caps = self.capabilities().methods[method]

        if config.x_scale is not None:
            raise ValueError(f"{method} does not support x_scale")
        if config.derivative_mode is not None and config.derivative_mode not in caps.derivative_modes:
            raise ValueError(f"{method} does not support derivative_mode={config.derivative_mode!r}")
        if config.loss not in caps.losses:
            raise ValueError(f"{method} does not support loss={config.loss!r}")

        request = {
            "problem_id": problem.id,
            "method": method,
            "x0": np.asarray(problem.starts[start], dtype=float).tolist(),
            "loss": _LOSS_TO_TRUST[config.loss],
        }
        if config.max_iter is not None:
            request["max_iter"] = config.max_iter
        if config.tolerance is not None:
            request["tolerance"] = config.tolerance
        if config.bounds is not None:
            lower, upper = config.bounds
            request["bounds"] = [np.asarray(lower, dtype=float).tolist(), np.asarray(upper, dtype=float).tolist()]
        if config.derivative_mode is not None:
            request["derivative_mode"] = config.derivative_mode

        response = _run_harness(request)

        x_final = response["x_final"]
        dist_to_opt = cost_gap = None
        if x_final is not None:
            optimum = problem.optima[0]
            dist_to_opt = float(np.linalg.norm(np.array(x_final) - optimum.x_star))
            cost_gap = float(response["cost_final"] - optimum.cost_star)

        return RunResult(
            problem_id=problem.id,
            backend=self.name,
            method=method,
            start=start,
            x_final=x_final,
            cost_final=response["cost_final"],
            dist_to_opt=dist_to_opt,
            cost_gap=cost_gap,
            grad_norm_final=response["grad_norm_final"],
            status=_STATUS[response["status"]],
            n_iter=response["n_iter"],
            n_feval=response["n_feval"],
            # trust's Eval convention returns the residual and Jacobian from
            # the same call; every counted evaluation is both at once.
            n_jeval=response["n_feval"],
            n_heval=response["n_heval"],
            trace=None,
            timing=None,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
