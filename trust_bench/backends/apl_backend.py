import atexit
import json
import re
import select
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from trust_bench.core.backend import Backend, Capabilities, MethodCapabilities
from trust_bench.core.provenance import capture, harness_git_sha
from trust_bench.core.result import RunResult, RunStatus
from trust_bench.core.timing import N_REPS, WARMUP, summarize

_SESSION_SCRIPT = Path(__file__).resolve().parents[2] / "backends_ext" / "apl" / "run_session.sh"

# Dyalog's interpreter is single-threaded for this workload (Newton.aplo/
# Min.aplo don't parallelise) - nothing to actively pin, unlike scipy's
# BLAS-backed NumPy calls, so this is a recorded fact, not a knob set.
_THREAD_COUNT = 1

_STATUS = {
    "CONVERGED": RunStatus.CONVERGED,
    "MAX_ITER": RunStatus.MAX_ITER,
    "FAILED": RunStatus.FAILED,
    "STALLED": RunStatus.STALLED,
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
        "solve_ms": None,
    }


# One dyalogscript process, started lazily on first use and shared by
# every APLBackend instance and evaluate_problem() call in this process
# (a study's sweep(), a test file's own APLBackend(), trust-bench
# report's CLI - all funnel through this same module-level handle,
# never their own instance state), for the life of the process. Runs
# backends_ext/apl/session.dyalog via run_session.sh: a persistent
# request/response loop over stdin/stdout instead of a fresh subprocess
# per call - interpreter startup (3-9s, confirmed directly) is paid
# once per process, not once per solve().
_session: subprocess.Popen | None = None
_session_buffer = b""
_session_lock = threading.Lock()


def _start_session() -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", str(_SESSION_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def _kill_session() -> None:
    global _session, _session_buffer
    if _session is not None:
        _session.kill()
        _session.wait()
    _session = None
    _session_buffer = b""


def _read_until_cr(proc: subprocess.Popen, deadline: float) -> bytes | None:
    """Bytes up to (not including) the next \\r in proc's stdout, or
    None on timeout/EOF.

    dyalogscript's own ⎕← terminates every physical line with a bare
    \\r, not \\n (confirmed directly) - Python's text-mode,
    universal-newlines readline() cannot safely handle this without
    risking a deadlock (it must peek one more byte to rule out \\r\\n,
    and the interpreter is meanwhile blocked waiting for the next
    request, confirmed directly by reproducing the hang), so this
    reads raw bytes and splits on \\r itself. select bounds the wait so
    a hung request doesn't block forever, without adding
    threading/queue machinery for what is still a synchronous,
    one-request-at-a-time protocol.
    """
    global _session_buffer
    while b"\r" not in _session_buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            return None
        chunk = proc.stdout.read(65536)
        if not chunk:
            return None
        _session_buffer += chunk
    line, _, _session_buffer = _session_buffer.partition(b"\r")
    return line


# dyalogscript's own hanging-indent width for a wrapped continuation
# segment (confirmed directly: every physical \r-terminated segment
# after the first one, in output past 32767 characters, starts with
# exactly 6 padding spaces that aren't part of the actual payload).
_WRAP_INDENT = 6


def _read_response(proc: subprocess.Popen, timeout: float) -> str | None:
    """One full response from proc's stdout, or None on timeout/EOF.

    ⎕← wraps any single output past 32767 characters into several
    physical \\r-terminated segments, each after the first padded with
    _WRAP_INDENT leading spaces (confirmed directly: a large Hessian at
    dimensionality(n=1000) triggers this, with no data loss across the
    wrap - only where the \\r lands and that padding). session.dyalog
    announces the payload's own length on its own short line first
    (always well under the wrap width), so the reader knows exactly how
    many payload characters to expect and can reassemble however many
    physical segments they arrive wrapped across.
    """
    deadline = time.monotonic() + timeout
    length_bytes = _read_until_cr(proc, deadline)
    if length_bytes is None:
        return None
    length = int(length_bytes)

    payload = bytearray()
    first_segment = True
    while len(payload) < length:
        segment = _read_until_cr(proc, deadline)
        if segment is None:
            return None
        if not first_segment:
            segment = segment[_WRAP_INDENT:]
        first_segment = False
        payload.extend(segment)
    return bytes(payload[:length]).decode()


def _send_request(request: dict) -> dict:
    """Round-trips one request through the shared session, starting or
    replacing it as needed. A dead session (crashed, or killed after a
    prior timeout) is silently replaced by a fresh one before this
    request is sent - only *this* request's own outcome is affected by
    whatever went wrong, not surfaced as a special exception the caller
    must handle differently from any other harness-reported ERROR.

    The request itself travels via a temp file, not directly as a
    stdin line: confirmed directly that ⍞ (APL's line-input primitive)
    silently mangles a piped, non-interactive line past roughly 1-2KB
    (e.g. a dimensionality(n=1000) request's own x0 array) - session.dyalog
    reads a short file *path* off stdin instead, well within any such
    limit, then reads the real request from that file the same way the
    one-shot run.dyalog already does. The response has no such
    limit (confirmed directly up to 20000 characters) and travels
    directly as a single ⎕← line.
    """
    global _session
    with _session_lock:
        if _session is None or _session.poll() is not None:
            _kill_session()
            _session = _start_session()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(request, f)
            request_path = f.name
        try:
            try:
                _session.stdin.write((request_path + "\n").encode())
                _session.stdin.flush()
            except (BrokenPipeError, OSError):
                _kill_session()
                return _timeout_result(f"harness did not complete within {_TIMEOUT_SECONDS}s")
            line = _read_response(_session, _TIMEOUT_SECONDS)
            if line is None:
                _kill_session()
                return _timeout_result(f"harness did not complete within {_TIMEOUT_SECONDS}s")
            return json.loads(line)
        finally:
            Path(request_path).unlink(missing_ok=True)


def _shutdown_session() -> None:
    global _session
    if _session is not None and _session.poll() is None:
        try:
            _session.stdin.write(b"\n")
            _session.stdin.flush()
            _session.stdin.close()
            _session.wait(timeout=5)
        except Exception:
            _session.kill()
    _session = None


atexit.register(_shutdown_session)


def evaluate_problem(problem_id: str, x) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Evaluate a problem's residual, Jacobian and Hessian at x via the APL
    harness's 'evaluate' mode. Used by the cross-language parity tests; not
    called by APLBackend.solve, which never needs a probe at an arbitrary
    point.
    """
    request = {"mode": "evaluate", "problem_id": problem_id, "x": np.asarray(x, dtype=float).tolist()}
    response = _send_request(request)
    if response["status"] != "OK":
        raise RuntimeError(response["message"])
    return response["residual"], response["jacobian"], response["hessian"]


@lru_cache(maxsize=1)
def _dyalog_version() -> str:
    # A fresh interpreter startup in its own right (confirmed directly,
    # ~3-7s) - previously invisible under the harness's own per-call
    # startup cost, now the dominant cost of every solve() call if not
    # cached, since the version can't change within one process's
    # lifetime.
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
            # trust's pscale is a fixed reparameterisation only: no
            # equivalent of scipy's adaptive x_scale="jac" (recomputed
            # every iteration from the current Jacobian), since
            # Newton.aplo's own damping by diag(H) already gives an
            # adaptive-scaling effect internally. The wrapper only
            # rescales a 2-item (value, derivative) return - lm's
            # (residual, jacobian) and BFGS's (cost, gradient) - not
            # trust-exact's 3-item (cost, hessian, gradient): a Hessian
            # needs outer-product scaling on both axes, not a column
            # scale, so silently applying pscale there would be a
            # silently wrong answer rather than an unsupported one.
            if isinstance(config.x_scale, str):
                raise ValueError(f"{method} does not support x_scale={config.x_scale!r}")
            if method == "trust-exact":
                raise ValueError(f"{method} does not support x_scale")
        if config.f_scale is not None:
            # trust's Loss namespace bakes in a fixed per-loss tuning
            # constant (Loss.apln: huber=1.345, cauchy=2.385) and
            # Min.aplo's L function further auto-scales it by a
            # MAD-based robust sigma recomputed every call - there is no
            # per-request knob to override either, so a silent no-op
            # would misrepresent what was actually run.
            raise ValueError(f"{self.name} does not support f_scale")
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
        if config.x_scale is not None:
            request["pscale"] = np.asarray(config.x_scale, dtype=float).tolist()

        # Warm-up run(s) discarded, then N_REPS measured repetitions -
        # docs/plans/trust-bench.md Section 7's timing policy. Each
        # repetition is just an ordinary _send_request round trip
        # through the already-persistent session (a few ms), not a
        # fresh subprocess spawn, so no special batching is needed here.
        # The persistent session only removes interpreter-startup cost
        # from each repetition; a round trip still pays JSON encoding/
        # decoding and IPC transfer, which is why solve_ms is measured
        # inside solve.dyalog itself (⎕AI around the Min call) rather
        # than by wall-clocking this round trip. A response with an
        # error stops the loop immediately rather than padding out the
        # remaining repetitions.
        samples = []
        for i in range(WARMUP + N_REPS):
            response = _send_request(request)
            if response["status"] == "ERROR":
                break
            if i >= WARMUP:
                samples.append(response["solve_ms"] / 1000.0)
        timing = (
            summarize(samples, warmup=WARMUP, n_reps=N_REPS, thread_count=_THREAD_COUNT)
            if len(samples) == N_REPS
            else None
        )

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
            timing=timing,
            config=config,
            provenance=self.environment(),
            harness_git_sha=harness_git_sha(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=response.get("message"),
        )
